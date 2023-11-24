import sys
sys.path.append(".")
from learning.training.train_supervised import Trainer
from data_io.train_data import file_exists
from data_io.models import load_model
from data_io.model_io import save_pytorch_model, load_pytorch_model
from data_io.weights import restore_pretrained_weights, save_pretrained_weights
from data_io.instructions import get_all_env_id_lists
from data_io.env import load_env_split
from parameters.parameter_server import initialize_experiment, get_current_parameters
from utils.logger import logger_info, setup_logger
from utils.seed_utils import setup_seed
#CHEN add
import warnings
warnings.filterwarnings('ignore')


# Supervised learning parameters
def train_supervised():
    initialize_experiment()
    setup_logger()
    setup = get_current_parameters()["Setup"]
    supervised_params = get_current_parameters()["Supervised"]
    num_epochs = supervised_params["num_epochs"]

    model, model_loaded = load_model()

    print("Loading data")
    train_envs, dev_envs, test_envs = get_all_env_id_lists(max_envs=setup["max_envs"])

    if "split_train_data" in supervised_params and supervised_params["split_train_data"]:
        split_name = supervised_params["train_data_split"]
        split = load_env_split()[split_name]
        train_envs = [env_id for env_id in train_envs if env_id in split]
        print("Using " + str(len(train_envs)) + " envs from dataset split: " + split_name)

    filename = "supervised_" + setup["model"] + "_" + setup["run_name"]
    start_filename = "tmp/" + filename + "_epoch_" + str(supervised_params["start_epoch"]-1)
    if supervised_params["start_epoch"] > 0:
        if file_exists(start_filename + ".pytorch"):
            logger_info("load model from epoch {}".format(supervised_params["start_epoch"]-1))
            load_pytorch_model(model, start_filename)
        else:
            print("Couldn't continue training. Model file doesn't exist at:")
            print(start_filename)
            exit(-1)

    if setup["restore_weights_name"]:
        restore_pretrained_weights(model, setup["restore_weights_name"], setup["fix_restored_weights"])

    trainer = Trainer(model, epoch=supervised_params["start_epoch"], name=setup["model"], run_name=setup["run_name"])

    logger_info("Num of epochs: " + str(num_epochs))
    logger_info("Beginning training...")
    best_test_loss = 1000
    best_epoch = 0
    for epoch in range(supervised_params["start_epoch"], num_epochs):
        train_loss = trainer.train_epoch(train_data=None, train_envs=train_envs, eval=False, epoch=epoch)

        trainer.model.correct_goals = 0
        trainer.model.total_goals = 0

        test_loss = trainer.train_epoch(train_data=None, train_envs=dev_envs, eval=True)

        print("GOALS: ", trainer.model.correct_goals, trainer.model.total_goals)

        if test_loss < best_test_loss:
            best_epoch = epoch
            best_test_loss = test_loss
            save_pytorch_model(trainer.model, filename)
            logger_info("Saved model in: " + filename)
        logger_info("Epoch " + str(epoch) + ", train_loss: " + str(train_loss) + ", test_loss: " + str(test_loss))
        save_pytorch_model(trainer.model, "tmp/" + filename + "_epoch_" + str(epoch))
        if hasattr(trainer.model, "save"):
            trainer.model.save(epoch)
        save_pretrained_weights(trainer.model, setup["run_name"])
    logger_info("Best epoch: " + str(best_epoch))
    logger_info("Min loss: " + str(best_test_loss))


if __name__ == "__main__":
    # setup_seed(42)
    train_supervised()
