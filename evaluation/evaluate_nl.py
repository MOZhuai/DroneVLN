import json
import os

import numpy as np

from evaluation.evaluate_base import EvaluateBase
from evaluation.results_t_landmark_side import ResultsLandmarkSide, K_RATE, K_AVG_DIST
from data_io.env import load_template, load_path, load_env_config
from data_io.instructions import get_all_instructions
from data_io.paths import get_results_path, get_results_dir
from utils.logging_summary_writer import LoggingSummaryWriter
from visualization import Presenter

DEFAULT_PASSING_DISTANCE = 100


def conbine_json(exist_result: dict, cur_result: dict):
    addable_dist = ['total_success', 'total_fail', 'total_segments', 'total_correct_landmarks', 'total_dist']
    for item in addable_dist:
        exist_result[item] += cur_result[item]
    exist_result['%success'] = exist_result['total_success'] / (exist_result['total_segments'] + 1e-28)
    exist_result['last_dist'] = cur_result['last_dist']
    exist_result['avg_dist'] = exist_result['total_dist'] / (exist_result['total_segments'] + 1e-28)
    exist_result['all_dist'].update(cur_result['all_dist'])
    all_dist_list = [i for it in list(exist_result['all_dist'].values()) for i in it]
    exist_result['median_dist'] = np.median(all_dist_list) if len(all_dist_list) > 0 else 0.0

    return exist_result


class DataEvalNL(EvaluateBase):

    def __init__(self, run_name="", save_images=True, entire_trajectory=True, custom_instr=None):
        super(EvaluateBase, self).__init__()
        self.train_i, self.test_i, self.dev_i, corpus = get_all_instructions()
        self.all_i = {**self.train_i, **self.test_i, **self.dev_i}
        self.passing_distance = DEFAULT_PASSING_DISTANCE
        self.results = ResultsLandmarkSide()
        self.presenter = Presenter()
        self.run_name = run_name
        self.save_images = save_images
        self.entire_trajectory = entire_trajectory
        self.custom_instr = custom_instr

    def evaluate_dataset(self, list_of_rollouts):
        self.results = ResultsLandmarkSide()
        for rollout in list_of_rollouts:
            if len(rollout) == 0:
                continue
            self.results += self.evaluate_rollout(rollout)

    def evaluate_rollout(self, rollout):
        last_sample = rollout[-1]
        env_id = last_sample["metadata"]["env_id"]
        seg_idx = last_sample["metadata"]["seg_idx"]
        set_idx = last_sample["metadata"]["set_idx"]

        # TODO: Allow multiple instruction sets / paths per env
        path = load_path(env_id)

        if self.entire_trajectory:
            path_end_idx = len(path) - 1
        else:
            # Find the segment end index
            path_end_idx = self.all_i[env_id][set_idx]["instructions"][seg_idx]["end_idx"]
            if path_end_idx > len(path) - 1:
                path_end_idx = len(path) - 1

        end_pos = np.asarray(last_sample["state"].get_pos())
        target_end_pos = np.asarray(path[path_end_idx])
        end_dist = np.linalg.norm(end_pos - target_end_pos)
        success = end_dist < DEFAULT_PASSING_DISTANCE

        if last_sample["metadata"]["pol_action"][3] > 0.5:
            who_stopped = "Policy Stopped"
        elif last_sample["metadata"]["ref_action"][3] > 0.5:
            who_stopped = "Oracle Stopped"
        else:
            who_stopped = "Veered Off"

        result = "Success" if success else "Fail"
        texts = [who_stopped, result, "run:" + self.run_name]

        print(seg_idx, result)

        if self.save_images:
            dir = get_results_dir(self.run_name, makedir=True)
            print("Results dir: ", dir)
            self.presenter.plot_paths(rollout, interactive=False, texts=texts, entire_trajectory=self.entire_trajectory)
            filename = os.path.join(dir, str(env_id) + "_" + str(set_idx) + "_" + str(seg_idx)) + "_" + result
            if self.custom_instr is not None:
                filename += "_" + last_sample["metadata"]["instruction"][:24] + "_" + last_sample["metadata"][
                                                                                          "instruction"][-16:]
            self.presenter.save_plot(filename)
            # self.save_results()

        return ResultsLandmarkSide(success, end_dist, env_id=env_id)

    def write_summaries(self, run_name, name, iteration):
        results_dict = self.get_results()
        writer = LoggingSummaryWriter(log_dir="runs/" + run_name, restore=True)
        if not K_AVG_DIST in results_dict:
            print("nothing to write")
            return
        writer.add_scalar(name + "/avg_dist_to_goal", results_dict[K_AVG_DIST], iteration)
        writer.add_scalar(name + "/success_rate", results_dict[K_RATE], iteration)
        writer.save_spied_values()

    def get_results(self):
        return self.results.get_dict()

    def save_results(self):
        # Write results dict
        path = get_results_path(self.run_name, makedir=True)
        print("path:", path)

        if os.path.isfile(path):
            with open(path, "r") as result_file:
                file_content = result_file.read()
        else:
            file_content = ''

        with open(path, "w") as result_file:
            cur_result = self.get_results()
            if file_content != '':
                exist_result = json.loads(file_content)
                cur_result = conbine_json(exist_result, cur_result)
            json.dump(cur_result, result_file)

        return cur_result
