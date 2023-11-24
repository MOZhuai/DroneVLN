import sys

import torch
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader

from data_io.instructions import get_all_instructions
from data_io.instructions import get_word_to_token_map
from learning.utils import get_n_params, get_n_trainable_params

from parameters.parameter_server import get_current_parameters
from utils.logger import logger_info

PROFILE = False


class Trainer:
    def __init__(
            self,
            model,
            state=None,
            epoch=0,
            name="",
            run_name="",
    ):
        _, _, _, corpus = get_all_instructions()
        self.token2word, self.word2token = get_word_to_token_map(corpus)

        self.params = get_current_parameters()["Training"]
        self.use_scheduler = self.params["use_scheduler"]
        self.batch_size = self.params['batch_size']
        self.weight_decay = self.params['weight_decay']
        self.optimizer = self.params['optimizer'].lower()
        self.num_loaders = self.params['num_loaders']
        self.lr = self.params['lr']
        self.decay_epoch = 10

        self.name = name

        n_params = get_n_params(model)
        n_params_tr = get_n_trainable_params(model)
        logger_info("Training Model:")
        logger_info("Number of model parameters: " + str(n_params))
        logger_info("Trainable model parameters: " + str(n_params_tr))

        self.model = model
        self.run_name = run_name
        logger_info("========================")
        logger_info("  Use scheduler: " + str(self.use_scheduler))
        if self.use_scheduler:
            logger_info("  Lr decay epoch: " + str(self.decay_epoch))
        logger_info("  Use optimizer: " + str(self.optimizer))
        logger_info("  learning rate: " + str(self.lr))
        logger_info("  Weight decay:  " + str(self.weight_decay))
        if self.optimizer == "adam":
            self.optim = optim.Adam(self.get_model_parameters(self.model), self.lr, weight_decay=self.weight_decay)
        elif self.optimizer == "sgd":
            logger_info("  Momentum:      " + str(self.params['momentum']))
            self.optim = optim.SGD(self.get_model_parameters(self.model), self.lr, weight_decay=self.weight_decay,
                                   momentum=self.params['momentum'])
        elif self.optimizer == "adamw":
            self.optim = optim.AdamW(self.get_model_parameters(self.model), self.lr, weight_decay=self.weight_decay)
        logger_info("========================")
        self.train_epoch_num = epoch
        self.train_segment = 0
        self.test_epoch_num = epoch
        self.test_segment = 0
        self.set_state(state)
        self.batch_num = 0

    def get_model_parameters(self, model):
        params_out = []
        skipped_params = 0
        for param in model.parameters():
            if param.requires_grad:
                params_out.append(param)
            else:
                skipped_params += 1
        return params_out

    def get_state(self):
        state = {}
        state["name"] = self.name
        state["train_epoch_num"] = self.train_epoch_num
        state["train_segment"] = self.train_segment
        state["test_epoch_num"] = self.test_epoch_num
        state["test_segment"] = self.test_segment
        return state

    def set_state(self, state):
        if state is None:
            return
        self.name = state["name"]
        self.train_epoch_num = state["train_epoch_num"]
        self.train_segment = state["train_segment"]
        self.test_epoch_num = state["test_epoch_num"]
        self.test_segment = state["test_segment"]

    def write_grad_summaries(self, writer, named_params, idx):
        for name, parameter in named_params:
            weights = parameter.data.cpu()
            mean_weight = torch.mean(weights)
            weights = weights.numpy()
            writer.add_histogram(self.model.model_name + "_internals" + "/hist_" + name + "_data", weights, idx,
                                 bins=100)
            writer.add_scalar(self.model.model_name + "_internals" + "/mean_" + name + "_data", mean_weight, idx)
            if parameter.grad is not None:
                grad = parameter.grad.data.cpu()
                mean_grad = torch.mean(grad)
                grad = grad.numpy()
                writer.add_histogram(self.model.model_name + "_internals" + "/hist_" + name + "_grad", grad, idx,
                                     bins=100)
                writer.add_scalar(self.model.model_name + "_internals" + "/mean_" + name + "_grad", mean_grad, idx)

    def write_grouped_loss_summaries(self, writer, losses, idx):
        pass

    def adjust_learning_rate(self, epoch=0):
        lr = self.lr * (0.1 ** (epoch // self.decay_epoch))
        for param_group in self.optim.param_groups:
            param_group['lr'] = lr

    def train_epoch(self, train_data=None, train_envs=None, eval=False, epoch=0):
        if eval:
            self.model.eval()
            inference_type = "eval"
            epoch_num = self.train_epoch_num
            self.test_epoch_num += 1
        else:
            self.model.train()
            inference_type = "train"
            epoch_num = self.train_epoch_num
            self.train_epoch_num += 1
            if self.use_scheduler:
                self.adjust_learning_rate(epoch)
            logger_info("current lr: " + str(set([param_group['lr'] for param_group in self.optim.param_groups])))

        dataset = self.model.get_dataset(data=train_data, envs=train_envs, dataset_name="supervised", eval=eval)

        if hasattr(dataset, "set_word2token"):
            dataset.set_word2token(self.token2word, self.word2token)

        dataloader = DataLoader(
            dataset,
            collate_fn=dataset.collate_fn,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_loaders,
            pin_memory=False,
            timeout=0,
            drop_last=False)

        num_samples = len(dataset)
        if num_samples == 0:
            print("DATASET HAS NO DATA!")
            return -1.0

        num_batches = int((num_samples + self.batch_size - 1) / self.batch_size)
        num_updates = epoch * 1

        second_order = hasattr(self.optim, 'is_second_order') and self.optim.is_second_order

        epoch_loss = 0
        count = 0
        epoch_map_loss = 0

        for batch in dataloader:

            if batch is None:
                # print("None batch!")
                continue

            # Zero gradients before each segment and initialize zero segment loss
            self.optim.zero_grad()

            batch_loss, map_loss = self.model.sup_loss_on_batch(batch, eval)

            # Backprop and step
            if not eval:
                batch_loss.backward(create_graph=second_order)
                self.batch_num += 1
                self.optim.step()

            torch.cuda.synchronize()

            # Get losses as floats
            epoch_loss += batch_loss.data.item()
            epoch_map_loss += map_loss.item() if map_loss else 0
            num_updates += 1
            count += 1

            sys.stdout.write(
                "\r" +
                " Batch:" + str(count) + " / " + str(num_batches) +
                " loss: " + str(batch_loss.item()) +
                " epoch loss: " + str(epoch_loss / (count + 1e-15)) +
                (" map_loss: " + str(epoch_map_loss / (count + 1e-15))[:7] if map_loss else "")
            )
            sys.stdout.flush()

            self.train_segment += 0 if eval else 1
            self.test_segment += 1 if eval else 0

        if hasattr(self.model, "write_eoe_summaries"):
            self.model.write_eoe_summaries(inference_type, epoch_num)

        print("")
        epoch_loss /= (count + 1e-15)

        if hasattr(self.model, "writer"):
            self.model.writer.add_scalar(self.name + "/" + inference_type + "_epoch_loss", epoch_loss, epoch_num)

        if epoch_map_loss > 0:
            logger_info(inference_type + " map_loss: " + str(epoch_map_loss / (count + 1e-15))[:7])

        return epoch_loss
