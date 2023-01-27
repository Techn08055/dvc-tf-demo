import tensorflow as tf
import os
import logging
from src.utils.all_utils import get_time_stamp

def get_VGG16_model(input_shape, model_path):
    model = tf.keras.applications.vgg16.VGG16(
        input_shape = input_shape,
        weights = "imagenet",
        include_top = False
    )
    model.save(model_path)
    logging.info(f"VGG16 modelsaved at:{model_path} ")
    return model

def prepare_model(model, CLASSES, freeze_all,freeze_till, learning_rate ):
    if freeze_all:
        for layer in model.layers:
            layer.trainable = False
    elif (freeze_till is not None) and (freeze_till > 1):
        for layer in model.layers[:-freeze_till]:
            layer.trainable = False

    # add our fully connected layers
    flatten_in = tf.keras.layers.Flatten()(model.output)
    prediction = tf.keras.layers.Dense(
        units = CLASSES, 
        activation= "softmax"
    )(flatten_in)

    foll_model = tf.keras.models.Model(
        inputs = model.input,
        outputs = prediction
    )
    foll_model.compile(
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate),
        loss = tf.keras.losses.CategoricalCrossentropy(),
        metrics = ['accuracy']
    )

    logging.info("custom model is compiled")

    return foll_model

def load_full_model(untrained_fill_model_path):
    model = tf.keras.models.load_model(untrained_fill_model_path)
    logging.info("Untrained model loaded")
    return model

def get_unique_path(train_model_dir_path, model_name = "model"):
    time_stamp = get_time_stamp(model_name)
    unique_model_name =  f"{time_stamp}_.h5"
    unique_model_path = os.path.join(train_model_dir_path, unique_model_name)

    return unique_model_path
