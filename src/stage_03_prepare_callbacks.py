from src.utils.all_utils import read_yaml, create_directory
from src.utils.models import get_VGG16_model, prepare_model
from src.utils.callbacks import create_and_save_tensorboard_callback, create_and_save_checkpoints_callback
import argparse
import pandas as pd 
import os
from tqdm import tqdm
import logging
import io
logging_str = "[%(asctime)s: %(levelname)s: %(module)s]: %(message)s"
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename=os.path.join(log_dir, 'running_logs.log'), level=logging.INFO, format=logging_str,
                    filemode="a")
def prepare_call_backs(config_path , params_path):
    config = read_yaml(config_path)
    params = read_yaml(params_path)

    artifacts = config["artifacts"]
    artifacts_dir = artifacts["ARTIFACTS_DIR"]
    tensorboard_log_dir = os,path.join(artifacts_dir, artifacts["TENSORBOARD_ROOT_LOG_DIR"])
    checkpoint_dir =  os,path.join(artifacts_dir, artifacts["checkpoints"])
    callbacks_dir = os,path.join(artifacts_dir, artifacts["callbacks"])

    create_directory([
        tensorboard_log_dir,
        checkpoint_dir,
        callbacks_dir
    ])

    create_and_save_tensorboard_callback(callbacks_dir, tensorboard_log_dir)
    create_and_save_checkpoints_callback(callbacks_dir, checkpoint_dir)


if __name__ == "__main__":
    args = argparse.ArgumentParser()

    args.add_argument("--config" , "-c", default = "config/config.yaml")
    args.add_argument("--params" , "-p", default = "params.yaml")
    
    parsed_args = args.parse_args()
    try:
        logging.info (">>>>stage three started")
        prepare_call_backs(config_path = parsed_args.config, params_path = parsed_args.params)
        logging.info ("stage three completed \n")

    except Exception as e:
        logging.exception(e)
        raise e