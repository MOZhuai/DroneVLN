# import stanfordnlp
#
# sentence = "The white and red flowers are blooming in the garden."
#
# # 加载英语模型
# nlp = stanfordnlp.Pipeline(processors='tokenize,pos,lemma,depparse')
#
# # 执行依存句法分析
# doc = nlp(sentence)
#
# # 提取名词短语和修饰词之间的依存关系
# phrases = []
# for sentence in doc.sentences:
#     for word in sentence.words:
#         if word.upos == 'NOUN':
#             head_word = sentence.words[word.head - 1]
#             if head_word.upos == 'ADJ':
#                 phrase = f"{head_word.text} {word.text}"
#                 phrases.append(phrase)
#
# print(phrases)  # ['white flowers', 'red flowers']
#
# # 提取单独的名词
# nouns = [word.text for sentence in doc.sentences for word in sentence.words if word.upos == 'NOUN' and word.head == 0]
# print(nouns)  # ['garden']
import stanfordnlp

# 加载英文模型
nlp = stanfordnlp.Pipeline(processors='tokenize,pos')

# 句子
sentence = "turn left so as to pass to the left of this structure     curving around to the right   as you approach the edge of the green area   you reach the end poiint"

# 处理句子
doc = nlp(sentence)

# 提取形容词名词组合
combinations = []
current_combination = []
for word in doc.sentences[0].words:
    if word.upos in ['ADJ', 'NOUN']:
        current_combination.append(word.text)
    else:
        if len(current_combination) > 1:
            combinations.append(' '.join(current_combination))
        elif len(current_combination) == 1:
            combinations.append(current_combination[0])
        current_combination = []

# 添加最后一个组合（如果有的话）
if len(current_combination) > 1:
    combinations.append(' '.join(current_combination))
elif len(current_combination) == 1:
    combinations.append(current_combination[0])

# 输出结果
print(combinations)

