# Vision-Language Navigation for Quadcopters with Conditional Transformer and Prompt-based Text Rephraser
### Intro
This is the code repository for the following paper:
"Vision-Language Navigation for Quadcopters with Conditional Transformer and Prompt-based Text Rephraser", Zhe Chen, Jiyi Li, Fumiyo Fukumoto, Peng Liu, Yoshimi Suzuki, Proceedings of the 2023 International Conference on ACM Multimedia Asia (MM Asia 2023)
### System Setup
1. The project is running on Ubuntu 18.04.
2. Install pytorch: 
`pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117`
3. Install other python library:  
`pip install numpy scipy opencv-python msgpack-rpc-python multiprocess PyUserInput yattag sympy PySimpleGUI transforms3d scikit-image pandas tensorboardX matplotlib sentence-transformers tqdm visdom torchsummary blobfile nibabel opencv-python scikit-learn scikit-image matplotlib pandas batchgenerators yacs`
4. Preparation for the data and simulator:
   1. Project Dir: "drif"
   2. Simulator: "DroneSimulator"
   3. dataset: "unreal_config_nl"
5. Configuration file: "drif/parameters/run_params/environments/corl_18.json"
   1. The location of simulator - "simulator_path": "(DroneSimulator extract dir)/DroneSimulator/LinuxNoEditor/MyProject5.sh"
   2. The location of simulator configuration file - "sim_config_dir": "/home/(your_username)/unreal_config/"
   3. The location of data set - "config_dir": point to path of "unreal_config_nl"

### Running Experiments
1. Use Oracle strategy to collect training data (already collected, DOWNLOAD):  
   `python mains/data_collect/collect_supervised_data.py corl_datacollect`
2. Trainning
   1. Working dir:  
   `cd script/spa_sbert`
   2. Stage 1 - Train to predict the position-visitation distributions:  
   `sh train_stage1.sh`
   3. Stage 2 - Train the policy network with the ground truth position-visitation distributions:  
   `sh train_stage2.sh`
   4. Stage 3 - Train the policy network with imitation learning DAggerFM:  
   `sh dagger.sh`
3. Evaluation:
   1. Working dir:  
   `cd script/spa_sbert`
   2. Evaluation:  
   `sh eval.sh`
   3. To evaluate the test and develop set, please modify the `eval_env_set` attribution in the configuration file "parameters/spa_sbert_eval.json".  
   ```json
    {
      "@include": [
        "spa_sbert_eval_base"
      ],
      "Setup":
      {
        "eval_env_set": "test/dev/triplet/gpt"
      }
    }
    ```
4. For models in other ablation experiments:
   1. For the model with condition-transformer: the training scripts are located at `script/spa`. The configuration files are located at `drif/parameters/run_params/spa/*.json`.
   2. For the model with sentence-bert: the training scripts are located at `script/sbert_only`. The configuration files are located at `drif/parameters/run_params/sbert_*.json`.

### Main Parameters
The parameters of the model are located at `spa_sbert_*.json` in the dictionary `parameters/run_params`.
1. `spatial_transformers` is used to configure the parameters for `condition-transformer`:
   1. `channels`: the number of input channels
   2. `n_heads`: the number of attention heads 
   3. `n_layers`: the number of layers
   4. `d_cond`: the length of instruction embedding
   5. `num_groups`: the setting of group-norm
2. `model_file` represents the name of the model to be loaded, and the default path is the path where the model is saved
3. The parameter `load_action_policy` in the file "spa_sbert_eval.json"
   1. If `load_action_policy=true`, you need to set `action_policy_file` as the path of the action prediction model, and load the model to overwrite this part of the parameters of the original model.
   2. If `load_action_policy=false`, only the parameters of the original model will be loaded.

### Others
1. The trained model are saved in the dictionary "unreal_config_nl/EVALUATED_MODEL".
2. The natural language instructions are stored in the file "unreal_config'nl/configs/tmp", and the JSON files in the folder represent different test instructions and environment configurations.
