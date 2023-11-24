import os
import sys
import logging

from parameters.parameter_server import get_current_parameters


def get_log_name():
    return get_current_parameters()["Setup"]["run_name"] + '_' + get_current_parameters()["Setup"]["model_type"]


def setup_logger():
    logger = logging.getLogger(get_log_name())
    logger.setLevel(logging.INFO)

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(levelname)s: %(asctime)s] %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    fh = logging.FileHandler(os.path.join("../dataset/unreal_config_nl/models", get_log_name() + '.txt'))
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


def logger_info(data):
    logger = logging.getLogger(get_log_name())
    logger.info(data)

