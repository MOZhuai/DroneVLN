### 项目配置
1. 项目运行在ubuntu18.04，更新版本的系统可能无法运行模拟器。
2. 安装pytorch, 项目中使用torch1.13:  
`pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117`
3. 安装其他python库:  
`pip install numpy scipy opencv-python msgpack-rpc-python multiprocess PyUserInput yattag sympy PySimpleGUI transforms3d scikit-image pandas tensorboardX matplotlib sentence-transformers tqdm visdom torchsummary blobfile nibabel opencv-python scikit-learn scikit-image matplotlib pandas batchgenerators yacs`
4. 准备项目必要文件:
   1. 项目文件夹: "drif"
   2. 无人机模拟器文件夹: "DroneSimulator"
   3. 数据集: "unreal_config_nl"
5. 配置文件 "drif/parameters/run_params/environments/corl_18.json"
   1. 无人机模拟器的位置 "simulator_path": "(DroneSimulator extract dir)/DroneSimulator/LinuxNoEditor/MyProject5.sh"
   2. 无人机模拟器配置文件的位置，路径只能在用户目录下 "sim_config_dir": "/home/(your_username)/unreal_config/"
   3. 数据集的位置 "config_dir": point to path of "unreal_config_nl"

### 运行(基础模型)
1. 使用oracle策略采集训练数据(已经采集好): `python mains/data_collect/collect_supervised_data.py corl_datacollect`
2. 训练:
   1. 进入脚本目录: `cd script/spa_sbert`
   2. 阶段1 - 训练预测访问分布: `sh train_stage1.sh`
   3. 阶段2 - 使用 ground truth 访问分布训练策略网络: `sh train_stage2.sh`
   4. 阶段3 - 使用 DAggerFM 训练策略网络: `sh dagger.sh`
3. 评估:
   1. 进入脚本目录: `cd script/spa_sbert`
   2. 运行: `sh eval.sh`
   3. 对于test和dev集的评估，通过修改配置文件"parameters/spa_sbert_eval.json"文件中的"eval_env_set"属性: 测试集(test)，开发集(dev)，LLM重写测试集(gpt)，三元组测试集(triplet)
4. 上述为"sentence-bert + condition-transformer"的训练评估步骤，对于其他的模型:
   1. 只有condition-transformer: 脚本文件存在"script/spa"中，对应的参数配置为"drif/parameters/run_params/spa/*.json"
   2. 只有sentence-bert: 脚本文件存在"script/sbert_only"中，对应的参数配置为"drif/parameters/run_params/sbert_*.json"

### 运行基础
### 主要参数
最终模型的配置参数存在文件夹 "parameters/run_params" 下的文件 "spa_sbert_*.json"中 (共五个文件)
1. "spatial_transformers" 用于配置 condition-transformer 的参数:
   1. "channels": 输入transformer的张量通道数
   2. "n_heads": 注意力头的数量
   3. "n_layers": transformer的层数
   4. "d_cond": instruction-embedding 的长度
   5. "num_groups": group-norm的参数设置
2. "model_file" 表示要训练阶段时需要加载的模型名称，默认的路径为模型保存的路径
3. "spa_sbert_eval.json" 文件下的 "load_action_policy" 表示是否加载其他(已经训练好的)动作预测模型
   1. 如果设置为true，需要配置 "action_policy_file" 为动作预测模型的路径，加载该模型覆盖原本模型的这部分参数
   2. 如果设置为false，则只会加载原本模型的参数
   3. 为了尽快查看模型训练的效果，可以加载提前训练好的动作预测模型。这样只需要进行一个阶段(阶段1)的训练而跳过剩下两个阶段的训练(因为阶段2-3都只训练动作预测网络)，已经训练好的动作模型存在"unreal_config_nl/models-origin/corl/action_gtr/map_to_action"中
### 其他
1. 所有训练好的模型存在 "unreal_config_nl/EVALUATED_MODEL" 文件夹下，备注了<map2action>的模型使用的是第一阶段训练的模型+已经训练好的动作模型(主要参数/第3.2节)
2. 虚拟环境下对应的指令数据文件存储在 "unreal_config_nl/configs/tmp"文件中，文件夹下的json文件表示不同的测试指令和对应的环境配置
