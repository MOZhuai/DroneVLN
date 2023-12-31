import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable

from imageio import imwrite
from PIL import Image
from data_io.model_io import load_pytorch_model
from data_io import env
from data_io.weights import enable_weight_saving
from learning.datasets.segment_dataset_simple import SegmentDataset
import learning.datasets.aux_data_providers as aup

from learning.inputs.common import empty_float_tensor, cuda_var
from learning.inputs.pose import Pose
from learning.inputs.sequence import none_padded_seq_to_tensor, len_until_nones
from learning.inputs.vision import standardize_image
from learning.datasets.aux_data_providers import get_top_down_ground_truth_static_global
from learning.datasets.aux_data_providers import get_top_down_ground_truth_static_current_global
from learning.modules.map_to_map.spatial_transformer import SpatialTransformer
from learning.modules.module_with_auxiliaries_base import ModuleWithAuxiliaries
from learning.modules.auxiliaries.class_auxiliary_2d import ClassAuxiliary2D
from learning.modules.auxiliaries.goal_auxiliary import GoalAuxiliary2D
from learning.modules.auxiliaries.path_auxiliary import PathAuxiliary2D
from learning.modules.auxiliaries.class_auxiliary import ClassAuxiliary
from learning.modules.auxiliaries.feature_reg_auxiliary import FeatureRegularizationAuxiliary2D
from learning.modules.goal_pred_criterion import GoalPredictionGoodCriterion
from learning.modules.action_loss import ActionLoss
from learning.modules.img_to_map.fpv_to_global_map import FPVToGlobalMap
from learning.modules.map_to_action.ego_map_to_action_triplet import EgoMapToActionTriplet
from learning.modules.map_to_action.cropped_map_to_action_triplet import CroppedMapToActionTriplet
from learning.modules.map_to_map.draw_start_pos import DrawStartPosOnGlobalMap
from learning.modules.map_to_map.ratio_path_predictor import RatioPathPredictor
from learning.modules.map_to_map.leaky_integrator_w import LeakyIntegratorGlobalMap
from learning.modules.map_to_map.identity_map_to_map import IdentityMapProcessor
from learning.modules.map_to_map.random_perturb import MapPerturbation
from learning.modules.map_to_map.lang_filter_map_to_map import LangFilterMapProcessor
from learning.modules.map_to_map.map_batch_fill_missing import MapBatchFillMissing
from learning.modules.map_to_map.map_batch_select import MapBatchSelect
from learning.modules.sentence_embeddings.sentence_bert import SentenceBert
from learning.modules.map_transformer_base import MapTransformerBase
from learning.modules.dbg_writer import DebugWriter
from learning.modules.sentence_embeddings.sentence_embedding_simple import SentenceEmbeddingSimple
from learning.modules.spatial_softmax_2d import SpatialSoftmax2d
from visualization import Presenter
from utils.logging_summary_writer import LoggingSummaryWriter
from utils.logger import logger_info
from learning.utils import save_tensor_as_img, get_viz_dir, draw_drone_poses

from learning.meters.moving_average import MovingAverageMeter
from learning.meters.meter_server import get_current_meters

from parameters.parameter_server import get_current_parameters
from ptest.test_affine import show_img, show_fpvs

# Set this to true to project the RGB image instead of feature map
IMG_DBG = False

RESNET_FACTOR = 4

MODEL_FPV = "fpv"
PVN_STAGE1_ONLY = "fpv_pretrain"
MODEL_FPV_SAVE_MAPS_ONLY = "fpv_savemaps"


class ModelTrajectoryTopDown(ModuleWithAuxiliaries):

    def __init__(self, run_name="", model_class=MODEL_FPV):

        super(ModelTrajectoryTopDown, self).__init__()
        self.model_name = "sm_trajectory" + str(model_class)
        self.model_class = model_class
        logger_info("Init model of type: " + str(model_class))
        self.run_name = run_name
        self.writer = LoggingSummaryWriter(log_dir="runs/" + run_name)
        self.setup = get_current_parameters()["Setup"]
        self.params = get_current_parameters()["ModelPVN"]
        self.aux_weights = get_current_parameters()["AuxWeights"]

        self.iter = nn.Parameter(torch.zeros(1), requires_grad=False)

        logger_info("MODEL_TYPE: " + self.setup["model_type"])

        # Auxiliary Objectives
        self.use_aux_class_features = self.params["aux"]["class_features"]
        self.use_aux_grounding_features = self.params["aux"]["grounding_features"]
        self.use_aux_class_on_map = self.params["aux"]["class_map"]
        self.use_aux_grounding_on_map = self.params["aux"]["grounding_map"]
        self.use_aux_goal_on_map = self.params["aux"]["goal_map"]
        self.use_aux_lang = self.params["aux"]["lang"]
        self.use_aux_traj_on_map = False
        self.use_aux_traj_on_map_ratio = self.params["aux"]["path"]
        self.use_aux_reg_map = self.params["aux"]["regularize_map"]

        self.do_perturb_maps = self.params["perturb_maps"] if self.model_class in [PVN_STAGE1_ONLY,
                                                                                   MODEL_FPV] else False
        logger_info("Perturbing maps: " + ("True" if self.do_perturb_maps else "False"))

        self.act_only = self.params["train_action_only"] if self.model_class in [MODEL_FPV] else False
        logger_info("Training only action: " + ("True" if self.act_only else "False"))

        self.detach_act_grad = self.params["detach_action_grad"] if self.model_class in [MODEL_FPV] else False
        logger_info("Detaching action gradients: " + ("True" if self.detach_act_grad else "False"))

        # Path-pred FPV model definition
        # --------------------------------------------------------------------------------------------------------------

        self.num_feature_channels = self.params["feature_channels"]

        self.num_map_channels = self.params["pathpred_in_channels"]

        self.img_to_features_w = FPVToGlobalMap(
            source_map_size=self.params["global_map_size"],
            world_size_px=self.params["world_size_px"],
            world_size=self.params["world_size_m"],
            res_channels=self.params["resnet_channels"],
            map_channels=self.params["feature_channels"],
            img_w=self.params["img_w"],
            img_h=self.params["img_h"],
            cam_h_fov=self.params["cam_h_fov"],
            img_dbg=IMG_DBG,
            model_type=self.setup["model_type"],
            cot_net=self.setup["cot_folder"] if "cot_folder" in self.setup else None
        )

        self.map_accumulator_w = LeakyIntegratorGlobalMap(source_map_size=self.params["global_map_size"],
                                                          world_in_map_size=self.params["world_size_px"])

        # Pre-process the accumulated map to do language grounding if necessary - in the world reference frame
        if self.setup["model_type"] in ["spa", "cot_spa", "spa_resx", "start_aux_spa"]:
            logger_info("Use SpatialTransformer")
            self.map_processor_a_w = SpatialTransformer(
                channels=self.params["spatial_transformers"]["channels"],
                n_heads=self.params["spatial_transformers"]["n_heads"],
                n_layers=self.params["spatial_transformers"]["n_layers"],
                d_cond=self.params["emb_size"],
                num_groups=self.params["spatial_transformers"]["num_groups"],
                dropout=self.params["spatial_transformers"]["dropout"]
            )
        elif self.use_aux_grounding_on_map and not self.use_aux_grounding_features:
            logger_info("Use LangFilterMapProcessor")
            self.map_processor_a_w = LangFilterMapProcessor(
                source_map_size=self.params["global_map_size"],
                world_size=self.params["world_size_px"],
                embed_size=self.params["emb_size"],
                in_channels=self.params["feature_channels"],
                out_channels=self.params["relevance_channels"],
                spatial=False, cat_out=True)
        else:
            logger_info("Use IdentityMapProcessor")
            self.map_processor_a_w = IdentityMapProcessor(source_map_size=self.params["global_map_size"],
                                                          world_size=self.params["world_size_px"])

        if self.params["condition_on_start_pos"]:
            self.map_processor_a_w2 = DrawStartPosOnGlobalMap(self.params["global_map_size"],
                                                              self.params["world_size_px"])
        else:
            self.map_processor_a_w2 = IdentityMapProcessor(source_map_size=self.params["global_map_size"],
                                                           world_size=self.params["world_size_px"])

        ratio_prior_channels = self.params["feature_channels"] + (1 if self.params["condition_on_start_pos"] else 0)

        logger_info("map to act channels: " + str(self.params["map_to_act_channels"]))
        # Process the global accumulated map
        self.map_processor_b_r = RatioPathPredictor(
            prior_channels=ratio_prior_channels,
            posterior_channels=self.params["pathpred_in_channels"],
            pred_channels=self.params["map_to_act_channels"],
            emb_size=self.params["emb_size"],
            source_map_size=self.params["local_map_size"],
            world_size=self.params["world_size_px"],
            compute_prior=self.params["compute_prior"],
            use_prior=self.params["use_prior_only"],
            l2=self.params["path_loss_function"] == "l2",
            ch_policy=self.params["channel_change_method"] if "channel_change_method" in self.params else "normal"
        )

        logger_info("UNet Channels: " + str(self.num_map_channels))
        logger_info("Feature Channels: " + str(self.num_feature_channels))

        enable_weight_saving(self.map_processor_b_r.unet_prior, "unet_prior", alwaysfreeze=self.act_only,
                             neverfreeze=False)
        enable_weight_saving(self.map_processor_b_r.unet_posterior, "unet_posterior", alwaysfreeze=self.act_only,
                             neverfreeze=False)

        enable_weight_saving(self.img_to_features_w.img_to_features, "img_to_features", alwaysfreeze=self.act_only)
        if self.use_aux_grounding_features:
            enable_weight_saving(self.img_to_features_w.lang_filter, "lang_filter_grounding",
                                 alwaysfreeze=self.act_only)
        if self.use_aux_grounding_on_map:
            enable_weight_saving(self.map_processor_a_w, "lang_filter_map_grounding",
                                 alwaysfreeze=self.act_only)

        self.second_transform = self.do_perturb_maps or self.params["predict_in_start_frame"]

        if self.second_transform:
            self.map_perturb = MapPerturbation(
                self.params["local_map_size"], self.params["world_size_px"])
            if self.use_aux_goal_on_map:
                raise Exception("Perturbed maps not supported together with map goal auxiliary")

        # Sentence Embedding
        if self.params["use_sbert"]:
            self.use_token = False
            self.sentence_embedding = SentenceBert(
                self.params["word_emb_size"],
                self.params["emb_size"],
                self.params["emb_layers"]
            )
        else:
            self.use_token = True
            self.sentence_embedding = SentenceEmbeddingSimple(
                self.params["word_emb_size"],
                self.params["emb_size"],
                self.params["emb_layers"]
            )

        self.map_transform_w_to_s = MapTransformerBase(source_map_size=self.params["global_map_size"],
                                                       dest_map_size=self.params["local_map_size"],
                                                       world_size=self.params["world_size_px"])

        self.map_transform_r_to_w = MapTransformerBase(source_map_size=self.params["local_map_size"],
                                                       dest_map_size=self.params["global_map_size"],
                                                       world_size=self.params["world_size_px"])

        # Batch select is used to drop and forget semantic maps at those timestaps that we're not planning in
        self.batch_select = MapBatchSelect()
        # Since we only have path predictions for some timesteps (the ones not dropped above), we use this to fill
        # in the missing pieces by reorienting the past trajectory prediction into the frame of the current timestep
        self.map_batch_fill_missing = MapBatchFillMissing(self.params["local_map_size"], self.params["world_size_px"])

        # Passing true to freeze will freeze these weights regardless of whether they've been explicitly reloaded or not
        enable_weight_saving(self.sentence_embedding, "sentence_embedding", alwaysfreeze=self.act_only)

        # Output an action given the global semantic map
        if self.params["map_to_action"] == "downsample2":
            self.map_to_action = EgoMapToActionTriplet(
                map_channels=self.params["map_to_act_channels"],
                map_size=self.params["local_map_size"],
                other_features_size=self.params["emb_size"])

        elif self.params["map_to_action"] == "cropped":
            self.map_to_action = CroppedMapToActionTriplet(
                map_channels=self.params["map_to_act_channels"],
                map_size=self.params["local_map_size"],
                manual=self.params["manual_rule"],
                path_only=self.params["action_in_path_only"],
                recurrence=self.params["action_recurrence"])

        self.spatialsoftmax = SpatialSoftmax2d()
        if self.params["train_action_from_dstar"]:
            self.gt_fill_missing = MapBatchFillMissing(self.params["local_map_size"], self.params["world_size_px"])

        if not self.params["load_action_policy"]:
            # Don't freeze the trajectory to action weights,
            # because it will be pre-trained during path-prediction training
            # and finetuned on all timesteps end-to-end
            enable_weight_saving(self.map_to_action, "map_to_action", alwaysfreeze=False, neverfreeze=True)

        # Auxiliary Objectives
        # --------------------------------------------------------------------------------------------------------------

        # We add all auxiliaries that are necessary. The first argument is the auxiliary name, followed by parameters,
        # followed by variable number of names of inputs. ModuleWithAuxiliaries will automatically collect these inputs
        # that have been saved with keep_auxiliary_input() during execution
        if self.use_aux_class_features:
            self.add_auxiliary(
                ClassAuxiliary2D(
                    "aux_class", None, self.params["feature_channels"],
                    self.params["num_landmarks"], 0, "fpv_features",
                    "lm_pos_fpv", "lm_indices"
                )
            )
        if self.use_aux_grounding_features:
            self.add_auxiliary(
                ClassAuxiliary2D(
                    "aux_ground", None, self.params["relevance_channels"],
                    2, 0, "fpv_features_g", "lm_pos_fpv", "lm_mentioned"
                )
            )
        if self.use_aux_class_on_map:
            self.add_auxiliary(
                ClassAuxiliary2D(
                    "aux_class_map", self.params["world_size_px"], self.params["feature_channels"],
                    self.params["num_landmarks"], 0, "map_s_w_select", "lm_pos_map_select", "lm_indices_select"
                )
            )
        if self.use_aux_grounding_on_map:
            self.add_auxiliary(
                ClassAuxiliary2D(
                    "aux_grounding_map", self.params["world_size_px"], self.params["relevance_channels"],
                    2, 0, "map_a_w_select", "lm_pos_map_select", "lm_mentioned_select"
                )
            )
        if self.use_aux_goal_on_map:
            self.add_auxiliary(
                GoalAuxiliary2D(
                    "aux_goal_map", self.params["goal_channels"],
                    self.params["world_size_px"], "map_b_w", "goal_pos_map"
                )
            )
        # CoRL model uses alignment-model groundings
        elif self.use_aux_lang:
            # one output for each landmark, 2 classes per output.
            # This is for finetuning, so use the embedding that's gonna be fine tuned
            self.add_auxiliary(
                ClassAuxiliary(
                    "aux_lang_lm_nl", self.params["emb_size"], 2,
                    self.params["num_landmarks"], "sentence_embed", "lang_lm_mentioned"
                )
            )
        loss_func = self.params["path_loss_function"]
        if self.use_aux_traj_on_map:
            self.add_auxiliary(PathAuxiliary2D("aux_path", loss_func, "map_b_r_select", "traj_gt_r_select"))

        if self.use_aux_traj_on_map_ratio:
            self.add_auxiliary(PathAuxiliary2D(
                "aux_path_prior", loss_func, "map_b_r_prior_select", "traj_gt_r_select"
            ))
            self.add_auxiliary(PathAuxiliary2D(
                "aux_path_posterior", loss_func, "map_b_r_posterior_select", "traj_gt_r_select"
            ))

        if self.use_aux_reg_map:
            self.add_auxiliary(FeatureRegularizationAuxiliary2D(
                "aux_regularize_features", None, "l1", "map_s_w_select", "lm_pos_map_select"
            ))

        self.goal_good_criterion = GoalPredictionGoodCriterion(ok_distance=3.2)
        self.goal_acc_meter = MovingAverageMeter(10)

        self.print_auxiliary_info()

        self.action_loss = ActionLoss()

        self.total_goals = 0
        self.correct_goals = 0

        self.visitation_ground_truth = False
        self.use_visitation_ground_truth = False

        self.env_id = None
        self.env_img = None
        self.seg_idx = None
        self.prev_instruction = None
        self.seq_step = 0
        self.get_act_start_pose = None
        self.gt_labels = None

    def cuda(self, device=None):
        ModuleWithAuxiliaries.cuda(self, device)
        self.sentence_embedding.cuda(device)
        self.map_accumulator_w.cuda(device)
        self.map_processor_a_w.cuda(device)
        self.map_processor_a_w2.cuda(device)
        self.map_processor_b_r.cuda(device)
        self.img_to_features_w.cuda(device)
        self.map_to_action.cuda(device)
        self.action_loss.cuda(device)
        self.map_batch_fill_missing.cuda(device)
        self.map_transform_w_to_s.cuda(device)
        self.map_transform_r_to_w.cuda(device)
        self.batch_select.cuda(device)
        if self.second_transform:
            self.map_perturb.cuda(device)
        if self.params["train_action_from_dstar"]:
            self.gt_fill_missing.cuda(device)
        return self

    def get_iter(self):
        return int(self.iter.data[0])

    def inc_iter(self):
        self.iter += 1

    def load_state_dict(self, state_dict, strict=False):
        super(ModelTrajectoryTopDown, self).load_state_dict(state_dict, strict)
        if self.params.get("load_image_feature"):
            load_pytorch_model(self.img_to_features_w, self.params["image_feature_file"])
            logger_info("Loaded image-to-features weights from file: " + self.params["image_feature_file"])

        # Override loading of weights to use the pre-trained action module
        if self.params["load_action_policy"]:
            load_pytorch_model(self.map_to_action, self.params["action_policy_file"])
            logger_info("Loaded map-to-action weights from file: " + self.params["action_policy_file"])

    def init_weights(self):
        self.img_to_features_w.init_weights()
        self.map_accumulator_w.init_weights()
        self.sentence_embedding.init_weights()
        self.map_to_action.init_weights()
        self.map_processor_a_w.init_weights()
        self.map_processor_b_r.init_weights()
        self.map_processor_a_w2.init_weights()

        if self.params.get("load_image_feature"):
            load_pytorch_model(self.img_to_features_w, self.params["image_feature_file"])
            logger_info("Loaded image-to-features weights from file: " + self.params["image_feature_file"])

        if self.params["load_action_policy"]:
            load_pytorch_model(self.map_to_action, self.params["action_policy_file"])
            logger_info("Loaded map-to-action weights from file: " + self.params["action_policy_file"])

    def reset(self):
        super(ModelTrajectoryTopDown, self).reset()
        self.sentence_embedding.reset()
        self.img_to_features_w.reset()
        self.map_accumulator_w.reset()
        # self.map_processor_a_w.reset()
        self.map_processor_a_w2.reset()
        self.map_processor_b_r.reset()
        self.map_transform_w_to_s.reset()
        self.map_transform_r_to_w.reset()
        self.map_batch_fill_missing.reset()
        if self.second_transform:
            self.map_perturb.reset()
        if self.params["train_action_from_dstar"]:
            self.gt_fill_missing.reset()
        self.prev_instruction = None
        self.get_act_start_pose = None

    def setEnvContext(self, context):
        print("Set env context to: " + str(context))
        self.env_id = context["env_id"]
        self.env_img = env.load_env_img(self.env_id, 256, 256)
        self.env_img = self.env_img[:, :, [2, 1, 0]]

    def set_ground_truth_visitation_d(self, visitation_distribution_g):
        if self.is_cuda:
            visitation_distribution_g = visitation_distribution_g.cuda()
        self.visitation_ground_truth = visitation_distribution_g
        self.use_visitation_ground_truth = True


    # def update_gtr_traj_with_current_position(self, position):
    #     import rollout.run_metadata as md
    #     m_size = self.params["local_map_size"]
    #     w_size = self.params["world_size_px"]
    #     self.gt_labels = self.maybe_cuda(
    #         get_top_down_ground_truth_static_current_global(
    #             md.ENV_ID, md.START_IDX, md.END_IDX, m_size, m_size, w_size, w_size, position
    #         )
    #     )

    def start_segment_rollout(self):
        import rollout.run_metadata as md
        m_size = self.params["local_map_size"]
        w_size = self.params["world_size_px"]
        self.gt_labels = get_top_down_ground_truth_static_global(
            md.ENV_ID, md.START_IDX, md.END_IDX, m_size, m_size, w_size, w_size)
        self.seg_idx = md.SEG_IDX
        self.gt_labels = self.maybe_cuda(self.gt_labels)
        if self.params["clear_history"]:
            self.start_sequence()

    def scale_images(self, images):
        if images.shape[0] == 72 and images.shape[1] == 144:
            return images
        return np.array(Image.fromarray(images).resize((144, 72)))

    def get_action(self, state, instruction):
        """
        Given a DroneState (from PomdpInterface) and instruction, produce a numpy 4D action (x, y, theta, pstop)
        :param state: DroneState object with the raw image from the simulator
        :param instruction: Tokenized instruction given the corpus
        :return:
        """

        with torch.no_grad():

            images_np_pure = state.image
            state_np = state.state

            # We might want to run the simulator at higher resultions in test time to get nice FPV videos.
            # If that's what we're doing, we need to scale the images back to the correct resolution
            images_np = self.scale_images(images_np_pure)
            images_np = standardize_image(images_np)
            image_fpv = Variable(none_padded_seq_to_tensor([images_np]))
            state = Variable(none_padded_seq_to_tensor([state_np]))
            # Add the batch dimension

            first_step = True
            if instruction == self.prev_instruction:
                first_step = False
            self.prev_instruction = instruction
            if first_step:
                self.get_act_start_pose = self.cam_poses_from_states(state[0:1])

            img_in_t = image_fpv
            img_in_t.volatile = True

            instr_len = None
            if self.use_token:
                instr_len = [len(instruction)] if instruction is not None else None
                instruction = torch.LongTensor(instruction).unsqueeze(0)
                instruction = cuda_var(instruction, self.is_cuda, self.cuda_device)

            state.volatile = True

            if self.is_cuda:
                if img_in_t is not None:
                    img_in_t = img_in_t.cuda(self.cuda_device)
                state = state.cuda(self.cuda_device)

            if self.model_class in [MODEL_FPV, MODEL_FPV_SAVE_MAPS_ONLY, PVN_STAGE1_ONLY]:
                step_enc = self.get_path_pos_encoding([True], None, self.seq_step)
                plan_now = [self.seq_step % self.params["plan_every_n_steps"] == 0 or first_step]
            else:
                step_enc = None
                plan_now = None

            start_pose = self.get_act_start_pose
            if self.is_cuda:
                start_pose = start_pose.cuda(self.cuda_device)

            self.seq_step += 1

            # This is for training the policy to mimic the ground-truth state distribution with oracle actions
            if self.params["run_action_from_dstar"] or self.params["write_figures"]:
                traj_gt_w = Variable(self.gt_labels)
                b_poses = self.cam_poses_from_states(state)
                # TODO: These source and dest should go as arguments to get_maps (in forward pass not params)
                transformer = MapTransformerBase(
                    source_map_size=self.params["global_map_size"],
                    world_size=self.params["world_size_px"],
                    dest_map_size=self.params["local_map_size"])
                self.maybe_cuda(transformer)
                transformer.set_maps(traj_gt_w, None)
                traj_gt_r, _ = transformer.get_maps(b_poses)
                self.clear_inputs("traj_gt_r_select")
                self.clear_inputs("traj_gt_w_select")
                self.keep_inputs("traj_gt_r_select", traj_gt_r)
                self.keep_inputs("traj_gt_w_select", traj_gt_w)

            action = self(
                img_in_t,
                state,
                instruction,
                instr_len=instr_len,
                plan=plan_now,
                pos_enc=step_enc,
                start_poses=start_pose,
                firstseg=[first_step]
            )

            if plan_now[0]:
                self.write_debug_data()
            if self.params["write_figures"]:
                self.save_viz(images_np_pure)

            output_action = action.squeeze().data.cpu().numpy()

            stop_prob = output_action[3]
            output_stop = 1 if stop_prob > self.params["stop_threshold"] else 0
            output_action[3] = output_stop

        return output_action

    def deterministic_action(self, action_mean, action_std, stop_prob):
        batch_size = action_mean.size(0)
        action = Variable(empty_float_tensor((batch_size, 4), self.is_cuda, self.cuda_device))
        action[:, 0:3] = action_mean[:, 0:3]
        action[:, 3] = stop_prob
        return action

    def sample_action(self, action_mean, action_std, stop_prob):
        action = torch.normal(action_mean, action_std)
        stop = torch.bernoulli(stop_prob)
        return action, stop

    # This is called before beginning an execution sequence
    def start_sequence(self):
        self.seq_step = 0
        self.reset()
        return

    def cam_poses_from_states(self, states):
        cam_pos = states[:, 9:12]
        cam_rot = states[:, 12:16]
        pose = Pose(cam_pos, cam_rot)
        return pose

    def save_viz(self, images_in):
        imwrite(get_viz_dir(self.env_id, self.seg_idx) + "fpv_" + str(self.seq_step) + ".png", images_in)
        features_cam = self.get_inputs_batch("fpv_features")[-1, 0, 0:3]
        save_tensor_as_img(features_cam, "F_c" + str(self.seq_step), self.env_id, self.seg_idx)
        feature_map_torch = self.get_inputs_batch("F_w")[-1, 0, 0:3]
        save_tensor_as_img(feature_map_torch, "F_w" + str(self.seq_step), self.env_id, self.seg_idx)
        coverage_map_torch = self.get_inputs_batch("M_w")[-1, 0, 0:3]
        save_tensor_as_img(coverage_map_torch, "M_w" + str(self.seq_step), self.env_id, self.seg_idx)
        semantic_map_torch = self.get_inputs_batch("map_s_w_select")[-1, 0, 0:3]
        save_tensor_as_img(semantic_map_torch, "S_w" + str(self.seq_step), self.env_id, self.seg_idx)
        relmap_torch = self.get_inputs_batch("map_a_w_select")[-1, 0, 0:3]
        save_tensor_as_img(relmap_torch, "R_w" + str(self.seq_step), self.env_id, self.seg_idx)

        trajpred_posterior = self.get_inputs_batch("map_b_w_posterior_select")[-1, 0, 0:3]
        trajpred_posterior = self.spatialsoftmax(trajpred_posterior.unsqueeze(0)).squeeze()
        save_tensor_as_img(trajpred_posterior, "D_u" + str(self.seq_step), self.env_id, self.seg_idx,
                           renorm_each_channel=True)
        trajpred_posterior_r = self.get_inputs_batch("map_b_r_posterior_select")[-1, 0, 0:3]
        trajpred_posterior_r = self.spatialsoftmax(trajpred_posterior_r.unsqueeze(0)).squeeze()
        save_tensor_as_img(trajpred_posterior_r, "D_u_r" + str(self.seq_step), self.env_id, self.seg_idx,
                           renorm_each_channel=True)
        trajpred_prior = self.get_inputs_batch("map_b_w_prior_select")[-1, 0, 0:3]
        trajpred_prior = self.spatialsoftmax(trajpred_prior.unsqueeze(0)).squeeze()
        save_tensor_as_img(trajpred_prior, "D_prior" + str(self.seq_step), self.env_id, self.seg_idx,
                           renorm_each_channel=True)

        dstar_w = self.get_inputs_batch("traj_gt_w_select")[-1, 0, 0:3]
        save_tensor_as_img(dstar_w, "dstar_w" + str(self.seq_step), self.env_id, self.seg_idx, renorm_each_channel=True)
        dstar_r = self.get_inputs_batch("traj_gt_r_select")[-1, 0, 0:3]
        save_tensor_as_img(dstar_r, "dstar_r" + str(self.seq_step), self.env_id, self.seg_idx, renorm_each_channel=True)

        drone_pos = self.get_inputs_batch("drone_poses")[-1, 0, 0:3]
        save_tensor_as_img(drone_pos, "pos" + str(self.seq_step), self.env_id, self.seg_idx)

        action = self.get_inputs_batch("action")[-1].data.cpu().squeeze().numpy()
        action_fname = get_viz_dir(self.env_id, self.seg_idx) + "action_" + str(self.seq_step) + ".png"
        Presenter().save_action(action, action_fname, "")

    def write_debug_data(self):
        writer = DebugWriter()

        if writer.should_write() and self.params["write_gifs"]:
            softmax = SpatialSoftmax2d()

            path_map = self.get_inputs_batch("map_b_w")[-1, 0]
            writer.write_img(path_map, "gif_overlaid", args={
                "world_size": self.params["world_size_px"], "name": "pathpred"
            })
            prior_map = softmax(self.get_inputs_batch("map_b_r_prior_select")[-1])[0]
            writer.write_img(prior_map, "gif_overlaid", args={
                "world_size": self.params["world_size_px"], "name": "prior"
            })
            posterior_map = softmax(self.get_inputs_batch("map_b_r_posterior_select")[-1])[0]
            writer.write_img(posterior_map, "gif_overlaid", args={
                "world_size": self.params["world_size_px"], "name": "posterior"
            })
            gnd_map = self.get_inputs_batch("map_a_w_select")[-1, 0]
            writer.write_img(gnd_map, "gif_overlaid", args={
                "world_size": self.params["world_size_px"], "name": "gnd"
            })
            fpv = self.get_inputs_batch("fpv")[-1, 0]
            writer.write_img(fpv, "gif", args={
                "world_size": self.params["world_size_px"], "name": "fpv"
            })

    def forward(self,
                images,
                states,
                instructions,
                has_obs=None,
                plan=None,
                save_maps_only=False,
                instr_len=None,
                pos_enc=None,
                noisy_poses=None,
                start_poses=None,
                firstseg=None
                ):
        """
        :param images: BxCxHxW batch of images (observations)
        :param states: BxK batch of drone states
        :param instructions: BxM LongTensor where M is the maximum length of any instruction
        :param has_obs: list of booleans of length B indicating whether the given sequence element has an observation
        :param plan:
        :param save_maps_only:
        :param pos_enc:
        :param noisy_poses:
        :param start_poses:
        :param firstseg:
        :return:
        """
        cam_poses = self.cam_poses_from_states(states)
        g_poses = None

        self.keep_inputs("fpv", images)

        # Calculate the instruction embedding
        if instructions is not None:
            if instr_len is not None:
                sent_emb = self.sentence_embedding(instructions, instr_len)
            else:
                sent_emb = self.sentence_embedding(instructions)
            self.keep_inputs("sentence_embed", sent_emb)
        else:
            sent_emb = self.sentence_embedding.get()

        if (not self.params["train_action_only"] or
            not self.params["train_action_from_dstar"] or
            not self.params["run_action_from_dstar"] or
            self.params["strain_stage2_with_model_output"]
        ) and not self.use_visitation_ground_truth:

            # Extract and project features onto the egocentric frame for each image
            features_w, coverages_w = self.img_to_features_w(images, cam_poses, sent_emb, self, show="")
            self.keep_inputs("F_w", features_w)
            self.keep_inputs("M_w", coverages_w)

            # Accumulate the egocentric features in a global map
            reset_mask = firstseg if self.params["clear_history"] else None
            maps_w = self.map_accumulator_w(features_w, coverages_w, add_mask=has_obs, reset_mask=reset_mask,
                                            show="acc" if IMG_DBG else "")
            map_poses_w = g_poses

            maps_w_select, map_poses_w_select, cam_poses_select, noisy_poses_select, start_poses_select, \
                sent_embeddings_select, pos_enc = self.batch_select(
                maps_w, map_poses_w, cam_poses, noisy_poses, start_poses,
                sent_emb, pos_enc, plan
            )

            maps_m_prior_select, maps_m_posterior_select = None, None

            # Only process the maps on planning timesteps
            if len(maps_w_select) > 0:
                self.keep_inputs("map_s_w_select", maps_w_select)

                # Create a figure where the drone is drawn on the map
                if self.params["write_figures"]:
                    self.keep_inputs("drone_poses", Variable(draw_drone_poses(cam_poses_select)))

                # Process the map via the two map_processors
                # Do grounding of objects in the map chosen to do so
                maps_w_select = self.map_processor_a_w(maps_w_select, sent_embeddings_select, map_poses_w_select)
                if isinstance(maps_w_select, tuple):
                    maps_w_select, map_poses_w_select = maps_w_select
                self.keep_inputs("map_a_w_select", maps_w_select)

                maps_w_select, map_poses_w_select = self.map_processor_a_w2(
                    maps_w_select, sent_embeddings_select, map_poses_w_select, cam_poses_select, show="draw_start")
                self.keep_inputs("map_a2_w_select", maps_w_select)

                if self.params["predict_in_start_frame"]:
                    s_poses_select = start_poses_select
                else:
                    s_poses_select = cam_poses_select

                self.map_transform_w_to_s.set_maps(maps_w_select, map_poses_w_select)
                maps_s_select, map_poses_s_select = self.map_transform_w_to_s.get_maps(s_poses_select)

                self.keep_inputs("map_a_s_select", maps_s_select)

                # Data augmentation for trajectory prediction
                # TODO: Should go inside trajectory predictor
                if self.do_perturb_maps:
                    assert noisy_poses_select is not None, "Noisy poses must be provided if we're perturbing maps"
                    maps_p_select, map_poses_p_select = self.map_perturb(maps_s_select, map_poses_s_select,
                                                                         noisy_poses_select)
                else:
                    maps_p_select, map_poses_p_select = maps_s_select, map_poses_s_select

                self.keep_inputs("map_a_s_perturbed_select", maps_s_select)

                # Include positional encoding for path prediction
                sent_embeddings_pp = sent_embeddings_select

                # Process the map via the two map_processors (e.g. predict the trajectory that we'll be taking)
                maps_p_select, maps_p_prior_select, maps_p_posterior_select, map_poses_p_select = \
                    self.map_processor_b_r(maps_p_select, sent_embeddings_pp, map_poses_p_select)

                # Un-perturb the maps - transform them to robot reference frame
                if self.second_transform:
                    # assert map_poses_clean_select is not None
                    map_poses_dirty_select = Pose(map_poses_p_select.position.clone(),
                                                  map_poses_p_select.orientation.clone())
                    maps_m_select, map_poses_m_select = self.map_perturb(maps_p_select, map_poses_dirty_select,
                                                                         cam_poses_select)
                    maps_m_prior_select, _ = self.map_perturb(maps_p_prior_select, map_poses_dirty_select,
                                                              cam_poses_select)
                    maps_m_posterior_select, _ = self.map_perturb(maps_p_posterior_select, map_poses_dirty_select,
                                                                  cam_poses_select)
                else:
                    maps_m_select = maps_p_select
                    map_poses_m_select = map_poses_p_select

                self.keep_inputs("map_b_r_select", maps_m_select)
                self.keep_inputs("map_b_r_prior_select", maps_m_prior_select)
                self.keep_inputs("map_b_r_posterior_select", maps_m_posterior_select)

                if self.params["write_figures"] or True:
                    self.map_transform_r_to_w.set_maps(maps_m_prior_select, map_poses_m_select)
                    maps_prior_w, _ = self.map_transform_r_to_w.get_maps(g_poses)
                    self.keep_inputs("map_b_w_prior_select", maps_prior_w)

                    self.map_transform_r_to_w.set_maps(maps_m_posterior_select, map_poses_m_select)
                    maps_posterior_w, _ = self.map_transform_r_to_w.get_maps(g_poses)
                    self.keep_inputs("map_b_w_posterior_select", maps_posterior_w)

            else:
                maps_m_select = None
                maps_posterior_w = None

            # If we're predicting the trajectory only on some timesteps, then for each timestep k, use the map from
            # timestep k if predicting on timestep k. otherwise use the map from timestep j - the last timestep
            # that had a trajectory prediction, rotated in the frame of timestep k.

            if self.model_class in [PVN_STAGE1_ONLY]:
                # If we're just pre-training the trajectory prediction, don't waste time on generating the missing maps
                maps_m = maps_m_select
                cam_poses = cam_poses_select
                sent_emb = sent_embeddings_select
            else:
                maps_m, map_poses_m = self.map_batch_fill_missing(maps_m_select, cam_poses, plan, show="")
                self.keep_inputs("map_b_r", maps_m)

            # Keep global maps for auxiliary objectives if necessary
            maps_b, _ = self.map_processor_b_r.get_maps(g_poses)
            self.keep_inputs("map_b_w", maps_b)

            # Output the final action given the processed map
            if self.detach_act_grad:
                maps_m = Variable(maps_m.data)
                sent_emb = Variable(sent_emb.data)

        # Predict action from ground-truth trajectory instead of the predicted trajectory
        # We are given ground truth visitation distributions to follow,
        # just transform those in the current frame and follow
        if self.use_visitation_ground_truth:
            self.map_transform_w_to_s.set_maps(self.visitation_ground_truth, None)
            maps_dstart_prob, _ = self.map_transform_w_to_s.get_maps(cam_poses)
            action_pred = self.map_to_action(maps_dstart_prob, sent_emb, fistseg_mask=firstseg)
        elif self.params["train_action_from_dstar"] \
                and self.params["run_action_from_dstar"] \
                and not self.params["strain_stage2_with_model_output"]:
            maps_dstar_select = self.get_inputs_batch("traj_gt_r_select")[:, 0]
            maps_dstart_prob, _ = self.gt_fill_missing(maps_dstar_select, cam_poses, plan, show="")
            action_pred = self.map_to_action(maps_dstart_prob, sent_emb, fistseg_mask=firstseg)
        else:
            action_pred = self.map_to_action(maps_m, sent_emb, fistseg_mask=firstseg)

        out_action = self.deterministic_action(action_pred[:, 0:3], None, action_pred[:, 3])
        self.keep_inputs("action", out_action)

        return out_action

    def maybe_cuda(self, tensor):
        if self.is_cuda:
            return tensor.cuda()
        else:
            return tensor

    def cuda_var(self, tensor):
        return cuda_var(tensor, self.is_cuda, self.cuda_device)

    def get_path_pos_encoding(self, plan_mask, b_metadata=None, step_num=0):
        trajectory_length = get_current_parameters()["Setup"]["trajectory_length"]
        step = 1.0 / float(trajectory_length)

        step_numbers = []
        last_seg = -1
        curr_step = 0
        if b_metadata is not None:
            for i in range(len(b_metadata)):
                if b_metadata[i]["seg_idx"] != last_seg:
                    curr_step = 0
                    last_seg = b_metadata[i]["seg_idx"]
                if plan_mask[i]:
                    step_numbers.append(curr_step)
                curr_step += 1
        else:
            step_numbers = [step_num]

        encoding = [s * step for s in step_numbers]
        encoding_t = torch.FloatTensor(encoding)
        encoding_t = self.maybe_cuda(encoding_t)
        encoding_t = Variable(encoding_t)
        return encoding_t

    # Forward pass for training
    def sup_loss_on_batch(self, batch, eval):
        action_loss_total = Variable(empty_float_tensor([1], self.is_cuda, self.cuda_device))

        if batch is None:
            print("Skipping None Batch")
            return action_loss_total

        if self.model_class in [MODEL_FPV, PVN_STAGE1_ONLY, MODEL_FPV_SAVE_MAPS_ONLY]:
            images = self.maybe_cuda(batch["images"])
        else:
            tdims = batch["top_down_images"].data.clone()
            images = Variable(self.maybe_cuda(tdims))

        instructions = batch["instr"] if isinstance(batch["instr"], list) else self.maybe_cuda(batch["instr"])
        instr_lengths = batch["instr_len"] if "instr_len" in batch else None
        states = self.maybe_cuda(batch["states"])
        actions = self.maybe_cuda(batch["actions"])

        # Auxiliary labels
        lm_pos_fpv = batch["lm_pos_fpv"]  # All object 2D coordinates in the first-person image
        lm_pos_map = batch["lm_pos_map"]  # All object 2D coordinates in the semantic map
        lm_indices = batch["lm_indices"]  # All object class indices
        lm_mentioned = batch["lm_mentioned"]  # 1/0 labels whether object was mentioned
        goal_pos_map = batch["goal_loc"]  # Goal location in semantic map
        lang_lm_mentioned = batch["lang_lm_mentioned"]  # 1/0 labels whether object was mentioned
        plan_mask = batch["plan_mask"]  # True for every timestep that we do visitation prediction
        firstseg_mask = batch["firstseg_mask"]  # True for every timestep that is a new instruction segment

        traj_ground_truth_select = self.maybe_cuda(batch["traj_ground_truth"])
        noisy_poses = batch["noisy_poses"]
        start_poses = batch["start_poses"]
        metadata = batch["md"]

        batch_size = images.size(0)
        count = 0

        for b in range(batch_size):
            self.reset()

            b_seq_len = len_until_nones(metadata[b])

            # Slice the data according to the sequence length
            b_metadata = metadata[b][:b_seq_len]
            b_images = images[b][:b_seq_len]
            b_instructions = instructions[b][:b_seq_len]
            b_instr_len = instr_lengths[b][:b_seq_len] if instr_lengths is not None else None
            b_states = states[b][:b_seq_len]
            b_actions = actions[b][:b_seq_len]
            b_lm_pos_fpv = lm_pos_fpv[b][:b_seq_len]
            b_lm_pos_map = lm_pos_map[b][:b_seq_len]
            b_lm_indices = lm_indices[b][:b_seq_len]
            b_lm_mentioned = lm_mentioned[b][:b_seq_len]
            b_goal_pos = goal_pos_map[b][:b_seq_len]
            b_traj_ground_truth_select = traj_ground_truth_select[b]
            b_lang_lm_mentioned = lang_lm_mentioned[b][:b_seq_len]
            b_noisy_poses = self.maybe_cuda(noisy_poses[b][:b_seq_len])
            b_start_poses = self.maybe_cuda(start_poses[b][:b_seq_len])

            b_plan_mask = plan_mask[b][:b_seq_len]
            b_obs_mask = [True for _ in range(b_seq_len)]
            b_plan_mask_t_cpu = torch.Tensor(b_plan_mask) == True
            b_plan_mask_t = self.maybe_cuda(b_plan_mask_t_cpu)
            b_firstseg_mask = firstseg_mask[b][:b_seq_len]

            b_lm_pos_map = [self.cuda_var(s.long()) if s is not None else None for s in b_lm_pos_map]
            b_lm_pos_fpv = [self.cuda_var((s / RESNET_FACTOR).long()) if s is not None else None for s in b_lm_pos_fpv]
            b_lm_indices = [self.cuda_var(s) if s is not None else None for s in b_lm_indices]
            b_lm_mentioned = [self.cuda_var(s) if s is not None else None for s in b_lm_mentioned]
            b_goal_pos = self.cuda_var(b_goal_pos)
            b_lang_lm_mentioned = self.cuda_var(b_lang_lm_mentioned)

            self.keep_inputs("lm_pos_fpv", b_lm_pos_fpv)
            self.keep_inputs("lm_pos_map", b_lm_pos_map)
            self.keep_inputs("lm_indices", b_lm_indices)
            self.keep_inputs("lm_mentioned", b_lm_mentioned)
            self.keep_inputs("goal_pos_map", b_goal_pos)
            self.keep_inputs("lang_lm_mentioned", b_lang_lm_mentioned)
            self.keep_inputs("traj_gt_global_select", b_traj_ground_truth_select)

            b_pos_enc = None
            if self.model_class in [MODEL_FPV, PVN_STAGE1_ONLY]:
                b_pos_enc = self.get_path_pos_encoding(b_obs_mask, b_metadata)

            # ----------------------------------------------------------------------------
            # Optional Auxiliary Inputs
            # ----------------------------------------------------------------------------
            if self.input_required("traj_gt_r_select"):
                b_poses = self.cam_poses_from_states(b_states)
                b_poses_select = b_poses[b_plan_mask_t]
                transformer = MapTransformerBase(
                    source_map_size=self.params["global_map_size"],
                    world_size=self.params["world_size_px"],
                    dest_map_size=self.params["local_map_size"])
                self.maybe_cuda(transformer)
                transformer.set_maps(b_traj_ground_truth_select, None)
                traj_gt_local_select, _ = transformer.get_maps(b_poses_select)
                self.keep_inputs("traj_gt_r_select", traj_gt_local_select)
                self.keep_inputs("traj_gt_w_select", b_traj_ground_truth_select)

            if self.input_required("lm_pos_map_select"):
                b_lm_pos_map_select = [lm_pos for i, lm_pos in enumerate(b_lm_pos_map) if b_plan_mask[i]]
                self.keep_inputs("lm_pos_map_select", b_lm_pos_map_select)
            if self.input_required("lm_indices_select"):
                b_lm_indices_select = [lm_idx for i, lm_idx in enumerate(b_lm_indices) if b_plan_mask[i]]
                self.keep_inputs("lm_indices_select", b_lm_indices_select)
            if self.input_required("lm_mentioned_select"):
                b_lm_mentioned_select = [lm_m for i, lm_m in enumerate(b_lm_mentioned) if b_plan_mask[i]]
                self.keep_inputs("lm_mentioned_select", b_lm_mentioned_select)

            if self.model_class in [PVN_STAGE1_ONLY]:
                # If we're just pre-training path,
                # the model will not output every action prediction,
                # but only action predictions on planning steps.
                b_actions = b_actions[b_plan_mask_t[:, np.newaxis].expand_as(b_actions)].view([-1, 4])

            # ----------------------------------------------------------------------------

            actions = self(
                b_images,
                b_states,
                b_instructions,
                instr_len=b_instr_len,
                has_obs=b_obs_mask,
                plan=b_plan_mask,
                pos_enc=b_pos_enc,
                firstseg=b_firstseg_mask,
                noisy_poses=b_noisy_poses,
                start_poses=b_start_poses
            )

            action_losses, _ = self.action_loss(b_actions, actions, batchreduce=False)

            # Check if the goal-prediction is good enough. If it is not, don't use this example for learning actions
            if self.model_class in [PVN_STAGE1_ONLY, MODEL_FPV] \
                    and not (self.params["train_action_from_dstar"] and self.params["run_action_from_dstar"]):
                maps_pp_in = self.get_inputs_batch("map_a_s_perturbed_select")
                pp_priors = self.get_inputs_batch("map_b_r_prior_select")
                pp_posteriors = self.get_inputs_batch("map_b_r_posterior_select")
                map_pathpreds = self.get_inputs_batch("map_b_r_select")
                traj_gts = self.get_inputs_batch("traj_gt_r_select")
                sm_globals = self.get_inputs_batch("map_s_w_select")
                good_goals = 0

                for i in range(1):
                    ppinmap = maps_pp_in[i]
                    pp_prior = self.spatialsoftmax(pp_priors[i])
                    pp_posterior = self.spatialsoftmax(pp_posteriors[i])
                    map_pathpred = map_pathpreds[i]
                    traj_gt = traj_gts[i]
                    sm_global = sm_globals[i]

                    iter = self.get_iter()
                    showstuff = iter % 60 == 0
                    # for finder : draw image show img plot
                    # if showstuff:
                    #     Presenter().show_image(ppinmap.data[0, 0:3], "map_a_r_gnd", torch=True, waitkey=1, scale=4)
                    #     Presenter().show_image(ppinmap.data[0, 3:6], "map_a_r_sm", torch=True, waitkey=1, scale=4)
                    #     Presenter().show_image(sm_global.data[0, 0:3], "sm_global", torch=True, waitkey=1, scale=8)
                    #     Presenter().show_image(pp_prior.data[0], "pp_prior", torch=True, waitkey=1, scale=4)
                    #     Presenter().show_image(pp_posterior.data[0], "pp_posterior", torch=True, waitkey=1, scale=4)

                    # We can't report goal-prediction accuracy if we don't have a goal-state channel.
                    if not self.params["action_in_path_only"]:
                        ok_goal = self.goal_good_criterion(map_pathpred, traj_gt, show="goal_pred" if showstuff else "")
                        correct_goal = 0
                        self.total_goals += 1
                        if not ok_goal.all():
                            # TODO: This does not align, because action losses include ALL actions, but
                            # predicitons only correspond to SELECT actions.
                            # perhaps track prediction accuracy during forward pass and fill_missing should
                            # produce an "action plausible" mask
                            if self.params["action_upd_correct_only"]:
                                action_losses[i] = action_losses[i] * 0
                        else:
                            self.correct_goals += 1
                            correct_goal = 1
                            good_goals += 1
                        self.goal_acc_meter.put(correct_goal)

            action_losses = self.action_loss.batch_reduce_loss(action_losses)
            action_loss = self.action_loss.reduce_loss(action_losses)

            action_loss_total = action_loss
            count += b_seq_len

        action_loss_avg = action_loss_total / (count + 1e-9)

        # Doing this in the end
        if self.params["run_auxiliaries"]:
            aux_losses = self.calculate_aux_loss(reduce_average=True)
            aux_loss = self.combine_aux_losses(aux_losses, self.aux_weights)
        else:
            aux_loss = Variable(empty_float_tensor([1], self.is_cuda, self.cuda_device))
            aux_losses = {}

        prefix = self.model_name + ("/eval" if eval else "/train")

        self.writer.add_dict(prefix, get_current_meters(), self.get_iter())
        self.writer.add_dict(prefix, aux_losses, self.get_iter())

        self.writer.add_scalar(prefix + "/goal_accuracy", self.goal_acc_meter.get(), self.get_iter())

        if self.model_class in [PVN_STAGE1_ONLY]:
            total_loss = aux_loss
        else:
            self.writer.add_scalar(prefix + "/action_loss", action_loss_avg.data.cpu().item(), self.get_iter())
            total_loss = action_loss_avg * self.aux_weights["action"] + aux_loss

        self.inc_iter()

        return total_loss, aux_losses["aux_path_posterior"] if "aux_path_posterior" in aux_losses else None

    def get_dataset(self, data=None, envs=None, dataset_name=None, eval=False):
        data_sources = [aup.PROVIDER_LM_POS_DATA, aup.PROVIDER_GOAL_POS, aup.PROVIDER_START_POSES]
        # If we're running auxiliary objectives, we need to include the data sources for the auxiliary labels

        # Adding these in this order will compute poses with added noise and compute trajectory ground truth
        # in the reference frame of these noisy poses

        if self.do_perturb_maps:
            print("PERTURBING MAPS!")
            data_sources.append(aup.PROVIDER_NOISY_POSES)
            # TODO: Think this through. Perhaps we actually want dynamic ground truth given a noisy start position
            if self.params["predict_with_current_position"]:
                data_sources.append(aup.PROVIDER_TRAJECTORY_GROUND_TRUTH_STATIC_CURRENT)
            elif self.params["predict_in_start_frame"]:
                data_sources.append(aup.PROVIDER_TRAJECTORY_GROUND_TRUTH_STATIC)
            else:
                data_sources.append(aup.PROVIDER_TRAJECTORY_GROUND_TRUTH_DYNAMIC_NOISY)
        else:
            print("NOT Perturbing Maps!")
            data_sources.append(aup.PROVIDER_NOISY_POSES)
            if self.params["predict_with_current_position"]:
                data_sources.append(aup.PROVIDER_TRAJECTORY_GROUND_TRUTH_STATIC_CURRENT)
            elif self.params["predict_in_start_frame"]:
                data_sources.append(aup.PROVIDER_TRAJECTORY_GROUND_TRUTH_STATIC)
            else:
                data_sources.append(aup.PROVIDER_TRAJECTORY_GROUND_TRUTH_DYNAMIC)

        data_sources.append(aup.PROVIDER_LANDMARKS_MENTIONED)

        templates = get_current_parameters()["Environment"]["templates"]
        if templates:
            data_sources.append(aup.PROVIDER_LANG_TEMPLATE)

        return SegmentDataset(data=data, env_list=envs, dataset_name=dataset_name, aux_provider_names=data_sources,
                              segment_level=True, use_token=self.use_token)
