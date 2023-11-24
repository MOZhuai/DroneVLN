import os
import torch
import torch.nn as nn
from .config import cfg, pop_unused_value
from .models import create_model, convert_splitbn_model
from .optim import create_optimizer
from .utils.logger import logger_info, setup_default_logging
from .scheduler import create_scheduler

def setup_env(folder=None, local_rank=0):
    if folder is not None:
        cfg.merge_from_file(os.path.join(folder, 'config.yaml'))
    cfg.root_dir = folder

    # setup_default_logging()

    world_size = 1
    rank = 0  # global rank
    cfg.distributed = torch.cuda.device_count() > 1

    if cfg.distributed:
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
    cfg.num_gpus = world_size

    pop_unused_value(cfg)
    cfg.freeze()

    if cfg.distributed:
        logger_info('Training in distributed mode with multiple processes, 1 GPU per process. Process %d, total %d.' % (rank, cfg.num_gpus))
    else:
        logger_info('Training with a single process on %d GPUs.' % cfg.num_gpus)
    torch.manual_seed(cfg.seed + rank)


def setup_model():
    model = create_model(
        cfg.model.name,
        pretrained=cfg.model.pretrained,
        num_classes=cfg.model.num_classes,
        drop_rate=cfg.model.drop,
        drop_connect_rate=None,  # DEPRECATED, use drop_path
        drop_path_rate=cfg.model.drop_path if 'drop_path' in cfg.model else None,
        drop_block_rate=cfg.model.drop_block if 'drop_block' in cfg.model else None,
        global_pool=cfg.model.gp,
        bn_tf=cfg.BN.bn_tf,
        bn_momentum=cfg.BN.bn_momentum if 'bn_momentum' in cfg.BN else None,
        bn_eps=cfg.BN.bn_eps if 'bn_eps' in cfg.BN else None,
        checkpoint_path=cfg.model.initial_checkpoint)


    if cfg.BN.split_bn:
        assert cfg.augmentation.aug_splits > 1 or cfg.augmentation.resplit
        model = convert_splitbn_model(model, max(cfg.augmentation.aug_splits, 2))
    model.cuda()
    return model


def setup_scheduler(optimizer, resume_epoch):
    lr_scheduler, num_epochs = create_scheduler(cfg, optimizer)
    start_epoch = 0
    if 'start_epoch' in cfg.solver:
        # a specified start_epoch will always override the resume epoch
        start_epoch = cfg.solver.start_epoch
    elif resume_epoch is not None:
        start_epoch = resume_epoch
    if lr_scheduler is not None and start_epoch > 0:
        lr_scheduler.step(start_epoch)

    return lr_scheduler, start_epoch, num_epochs


def load_CoT(config_folder=None):
    if config_folder is None:
        config_folder = "CoTNeXt-50-350epoch"
    print("========================")
    print("CoT Model Name:    " + config_folder)
    print("========================")
    config_path = "learning/modules/resnet/CoTNet/cot_experiments"
    config_folder = os.path.join(config_path, config_folder)
    setup_env(folder=config_folder, local_rank=1)
    model = setup_model()
    model.layer2 = nn.Identity()
    model.layer3 = nn.Identity()
    model.layer4 = nn.Identity()
    model.global_pool = nn.Identity()
    model.fc = nn.Identity()
    return model


def create_optims(model):
    # config_path = "learning/modules/resnet/CoTNet/cot_experiments"
    # config_folder = os.path.join(config_path, config_folder)
    # setup_env(folder=config_folder, local_rank=1)
    optimizer = create_optimizer(cfg, model)
    model_ema, resume_epoch, loss_scaler = None, None, None
    lr_scheduler, start_epoch, num_epochs = setup_scheduler(optimizer, resume_epoch)

    return optimizer, lr_scheduler, cfg