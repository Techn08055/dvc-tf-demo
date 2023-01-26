from src.utils.all_utils import read_yaml, create_directory
from src.utils.callbacks import create_and_save_tensorboard_callback, create_and_save_checkpoints_callback
import argparse
import pandas as pd 
import os
from tqdm import tqdm
import logging

logging_str = "[%(asctime)s: %(levelname)s: %(module)s]: %(message)s"
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename=os.path.join(log_dir, 'running_logs.log'), level=logging.INFO, format=logging_str,
                    filemode="a")

def train_model(config_path , params_path):
    config = read_yaml(config_path)
    params = read_yaml(params_path)

    artifacts = config["artifacts"]
    artifacts_dir = artifacts["ARTIFACTS_DIR"]

if __name__ == "__main__":
    args = argparse.ArgumentParser()

    args.add_argument("--config" , "-c", default = "config/config.yaml")
    args.add_argument("--params" , "-p", default = "params.yaml")
    
    parsed_args = args.parse_args()
    try:
        logging.info (">>>>stage four started")
        train_model(config_path = parsed_args.config, params_path = parsed_args.params)
        logging.info ("stage four completed \n")

    except Exception as e:
        logging.exception(e)
        raise e