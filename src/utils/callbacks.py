import tensorflow as tf
import os
import time
import logging
import joblib
from src.utils.all_utils import get_time_stamp


def create_and_save_tensorboard_callback(callbacks_dir, tensorboard_log_dir):
    unique_name = get_time_stamp("tb_log")

    tb_running_log_dir = os.path.join(tensorboard_log_dir, unique_name)
    tensorboard_callbacks = tf.keras.callbacks.TensorBoard(log_dir=tb_running_log_dir)
    tb_callback_filepath = os.path.join(callbacks_dir,"tensorboard_cb.cb")
    joblib.dump(tensorboard_callbacks, tb_callback_filepath)
    logging.info("tensorflow logs saved")

def create_and_save_checkpoints_callback(callbacks_dir, checkpoint_dir):
    checkpoint_file = os.path.join (checkpoint_dir, "ckpt_model.h5")
    checkpoints_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath= checkpoint_file,
        save_best_only= True,  
    )
    ckpt_callback_filepath = os.path.join(callbacks_dir,"checkpoint_cb.cb")
    joblib.dump(checkpoints_callback, ckpt_callback_filepath)
    logging.info("chekpoints logs saved")