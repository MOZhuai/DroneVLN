import numpy as np

from evaluation.results_base import EvaluationResults

K_N_SUCCESS = "total_success"
K_N_FAIL = "total_fail"
K_N_SEG = "total_segments"
K_N_LM = "total_correct_landmarks"
K_RATE = "%success"
K_RATE_LM = "%correct_landmark"
K_DIST = "total_dist"
K_AVG_DIST = "avg_dist"

K_LAST_DIST = "last_dist"

K_MEDIAN_DIST = "median_dist"
K_ALL_DIST = "all_dist"


class ResultsLandmarkSide(EvaluationResults):

    def __init__(self, success=None, end_dist=0, correct_landmark=False, env_id=-1):
        super(EvaluationResults, self).__init__()
        env_id = str(env_id)
        self.state = {
            K_N_SUCCESS: 1 if success else 0 if success is not None else 0,
            K_N_FAIL: 0 if success else 1 if success is not None else 0,
            K_N_SEG: 1 if success is not None else 0,
            K_RATE: 1.0 if success else 0.0 if success is not None else 0,
            K_N_LM: 1 if correct_landmark else 0,
            K_DIST: end_dist,
            K_LAST_DIST: end_dist
        }
        self.env_id = env_id
        self.metastate_distances = {}

    def __add__(self, past_results):
        self.state[K_N_SUCCESS] += past_results.state[K_N_SUCCESS]
        self.state[K_N_FAIL] += past_results.state[K_N_FAIL]
        self.state[K_N_SEG] += past_results.state[K_N_SEG]
        self.state[K_RATE] = self.state[K_N_SUCCESS] / (self.state[K_N_SEG] + 1e-28)

        self.state[K_DIST] += past_results.state[K_DIST]
        self.state[K_AVG_DIST] = self.state[K_DIST] / (self.state[K_N_SEG] + 1e-28)

        self.state[K_N_LM] = self.state[K_N_LM] + past_results.state[K_N_LM]
        self.state[K_RATE_LM] = self.state[K_N_LM] / (self.state[K_N_SEG] + 1e-28)

        if past_results.env_id in self.metastate_distances:
            self.metastate_distances[past_results.env_id].append(past_results.state[K_LAST_DIST])
        else:
            self.metastate_distances[past_results.env_id] = [past_results.state[K_LAST_DIST]]

        return self

    def get_dict(self):
        all_dist_list = []
        for value_list in self.metastate_distances.values():
            all_dist_list.extend(value_list)
        self.state[K_MEDIAN_DIST] = np.median(all_dist_list) if len(all_dist_list) > 0 else 0.0
        self.state[K_ALL_DIST] = self.metastate_distances
        return self.state
