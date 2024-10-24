import wandb
import json
import os
import matplotlib.pyplot as plt
import numpy as np
from keras_segmentation.models.unet import vgg_unet
from keras.callbacks import Callback
from keras import backend as K
from keras.models import load_model
from keras_segmentation.predict import model_from_checkpoint_path


def model_from_specific_checkpoint_path(checkpoints_path, specific_checkpoint_name):
    from .models.all_models import model_from_name
    specific_checkpoint_path = os.path.join(checkpoints_path, specific_checkpoint_name)
    assert (os.path.isfile(specific_checkpoint_path + "_config.json")), "Checkpoint not found."
    model_config = json.loads(open(specific_checkpoint_path + "_config.json", "r").read())
    model = model_from_name[model_config['model_class']](
        model_config['n_classes'], input_height=model_config['input_height'],
        input_width=model_config['input_width'])
    
    latest_weights = specific_checkpoint_path + ".h5"  # Assuming .h5 for weights
    assert (os.path.isfile(latest_weights)), "Weights file not found."
    
    print("Loaded weights from ", latest_weights)
    status = model.load_weights(latest_weights)

    if status is not None:
        status.expect_partial()

    return model

# tracking with wandb
wandb.init(
    name = "SCDD_20211104_predict_multiple_latest_checkpoint",
    project="scdd_segmentation_keras", 
    entity="ubix",
    config={
        "architecture": "vgg_unet",
        "dataset": "SCDD_20211104",
        "n_classes": 24,
        "input_height": 416,
        "input_width": 608,
    })

# test images and annotations path
test_image_path = "/SCDD-image-segmentation-keras/share/SCDD_20211104/images_test"
test_annotation_dir ="/SCDD-image-segmentation-keras/share/SCDD_20211104/masks_coded_test"

# Checkpoint path
checkpoint_path ="/SCDD-image-segmentation-keras/checkpoint/SCDD_20211104_vgg_unet/"
specific_checkpoint_name = ".0.index"

# Paths to save prediction
prediction_output_dir = "/SCDD-image-segmentation-keras/share/multi_predictions_SCDD_20211104"
if not os.path.exists(prediction_output_dir):
    os.makedirs(prediction_output_dir)

# Load the model
if os.path.exists(checkpoint_path):
    model = model_from_checkpoint_path(checkpoint_path)
else:
    raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")
    
# Predict segmentation
predictions = model.predict_multiple(
    inp_dir=test_image_path,
    out_dir=prediction_output_dir
)

for out_frame in os.listdir(prediction_output_dir):
    # Log the image with its index as part of the caption
    wandb.log({f"predictions/{out_frame}": wandb.Image(os.path.join(prediction_output_dir, out_frame), caption=f"Prediction for {out_frame}")})



# evaluating the model 
evaluation_result= model.evaluate_segmentation( inp_images_dir= test_image_path , annotations_dir= test_annotation_dir)

print(evaluation_result)

# Log evaluation results
wandb.log({"frequency_weighted_IU": evaluation_result['frequency_weighted_IU'], 
            "mean_IU": evaluation_result['mean_IU'], 
            "class_wise_IU": evaluation_result['class_wise_IU'], 
            })

# Finish the WandB run
wandb.finish()