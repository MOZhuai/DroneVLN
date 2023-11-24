import json
import os
import re
import random
import stanfordnlp
from stanfordnlp import Document
from stanfordnlp.pipeline.doc import Word
from env_config.generation.util.word_split import *


def random_select(source_path, out_path, replace=False, n=426):
    if os.path.exists(out_path):
        if replace:
            print("Replace existing file:", out_path)
        else:
            print("Output file exist")
            return

    if not os.path.exists(source_path):
        print("Source file not found:", source_path)
        exit(-1)

    with open(os.path.join(source_path), 'r') as source_fp:
        content = source_fp.read()
        instructions = json.loads(content)
        select_data = {}
        select_keys = []
        total_lenth = 0
        while total_lenth < n:
            random_env = random.choice(list(instructions.keys()))
            if random_env in select_keys:
                continue
            select_keys.append(random_env)
            select_data[random_env] = instructions[random_env]
            total_lenth += len(instructions[random_env][0]["instructions"])
        print("select data number:", total_lenth)

    with open(out_path, 'w') as out_fp:
        json.dump(select_data, out_fp)


def get_simple_instr(docu: Document):
    # tokens = [word.text for word in doc.sentences[0].words]
    reject_words = set()
    assert isinstance(docu.sentences[0].words[0], Word), "word type error"
    new_sent = []
    for word in docu.sentences[0].words:
        if (word.upos in ['NOUN', 'ADJ', 'PROPN'] or
            word.text.lower() in DIRECTION + ACTION + QUANTITY + SPECIAL_WORDS + COLOR + NOUN) \
                and word.text.lower() not in REJECT:
            new_sent.append(word.text)
        else:
            reject_words.add(word.text)
    return ' '.join(new_sent), reject_words


def get_simple_tuple(instruction, tuples=None):
    if tuples is None:
        tuples = []
    # loop each split_instruction
    for rule in RULES:
        match = re.search(rule[0], instruction)
        if match:
            # find match before this segment instruction
            if match.regs[0][0] != 0:
                tuples = get_simple_tuple(instruction[:match.regs[0][0]], tuples)

            # add current match into tuples
            if len(match.regs) > 0:
                for r in rule[1:]:
                    triple_tuple = []
                    for pos in r:
                        if pos is None:
                            triple_tuple.append('-')
                        elif isinstance(pos, str):
                            triple_tuple.append(pos)
                        elif isinstance(pos, int):
                            digits = [int(digit) for digit in str(pos)]
                            if len(digits) == 1:
                                triple_tuple.append(instruction[match.regs[pos][0]:match.regs[pos][1]])
                            elif digits[-1] == 0:
                                triple_tuple.append(
                                    "right"
                                    if instruction[match.regs[digits[0]][0]:match.regs[digits[0]][1]] == "left"
                                    else "left"
                                )
                            elif isinstance(r[0], str) and r[0] == "between" and 10 < r[1] < 100:
                                triple_tuple.append(" and ".join([
                                    instruction[match.regs[digit][0]:match.regs[digit][1]]
                                    for digit in digits
                                ]))
                            else:
                                triple_tuple.append(' '.join([
                                    instruction[match.regs[digit][0]:match.regs[digit][1]]
                                    for digit in digits
                                ]))
                    tuples.append(tuple(triple_tuple))

            # find match after this segment instruction
            if match.regs[0][1] != len(instruction):
                tuples = get_simple_tuple(instruction[match.regs[0][1]:], tuples)
            break
    return tuples


def get_clear_instructions_json(path):
    assert path[-5:] == ".json"
    res = {}
    with open(path, 'r') as fp:
        json_str = fp.read()
        inst = json.loads(json_str)
        for idx, item in enumerate(inst):
            res[inst[item][0]["env"]] = []
            cur_inst = inst[item][0]["instructions"]
            for instruction in cur_inst:
                res[inst[item][0]["env"]].append(instruction["instruction"])
    with open(path[:-5] + "_clear.json", 'w') as fp:
        json.dump(res, fp)


def str_tuple_list(tuples):
    return ', '.join('({}, {}, {})'.format(*tup) for tup in tuples)


def main():
    # trans tuples to str
    str_tuple = True
    config_dir = "/home/g21tka18/mygit/dataset/unreal_config_nl/configs/tmp"
    reject_words = set()
    instr_num = 0
    unchanged_cnt = 0
    unmatched_list = []
    origin_path = os.path.join(config_dir, "test.json")
    sep_origin_path = os.path.join(config_dir, "sep_origin_test.json")
    sep_simple_path = os.path.join(config_dir, "sep_simple_test.json")
    unmatched_path = os.path.join(config_dir, "unmatched.json")
    # sep_to_gpt_path = os.path.join(config_dir, "sep_to_gpt_test.json")
    reject_file_log = os.path.join(config_dir, "reject_log.json")

    # random choose env -> seg_test_data
    random_select(origin_path, sep_origin_path, replace=False, n=426)

    # init the stanfordnlp
    # nlp = stanfordnlp.Pipeline(processors='tokenize,pos')

    with open(sep_origin_path, 'r') as fp:
        json_str = fp.read()
        inst = json.loads(json_str)
        for idx, item in enumerate(inst):
            cur_inst = inst[item][0]["instructions"]
            for instruction in cur_inst:
                tuples_temp = get_simple_tuple(instruction["instruction"])
                if len(tuples_temp) == 0:
                    unmatched_list.append(instruction["instruction"])
                    unchanged_cnt += 1
                instr_num += 1
                instruction["origin_instruction"] = instruction["instruction"]
                instruction["simple_instruction"] = tuples_temp if not str_tuple else str_tuple_list(tuples_temp)
                instruction["instruction"] = "None"

    with open(sep_simple_path, 'w') as fp:
        json.dump(inst, fp)

    with open(unmatched_path, 'w') as fp:
        json.dump(unmatched_list, fp)

    print("instruction num:", instr_num)
    print("  unchanged num:", unchanged_cnt)
    #             doc = nlp(instruction["instruction"])
    #             simple_instruction, cur_reject_words = get_simple_instr(doc)
    #             reject_words.update(cur_reject_words)
    #             instr_num += 1
    #             if instr_num % 50 == 0:
    #                 print(instr_num, ":", instruction["instruction"], "\n        ---> ", simple_instruction)
    #             instruction["origin_instruction"] = instruction["instruction"]
    #             instruction["instruction"] = simple_instruction
    #
    # print("unchanged instruction number: ", unchanged_cnt)
    # print("reject words number: ", len(reject_words))
    #
    # with open(sep_simple_path, 'w') as nfp:
    #     json.dump(inst, nfp)
    #
    # reject_words = list(reject_words)
    # reject_words.sort()
    # with open(reject_file_log, 'w') as rjfp:
    #     json.dump(reject_words, rjfp)


if __name__ == '__main__':
    # get_clear_instructions_json("/home/g21tka18/mygit/dataset/unreal_config_nl/configs/tmp/"+"sep_origin_test.json")
    main()
