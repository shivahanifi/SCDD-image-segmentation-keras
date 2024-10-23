import wandb
import os
import matplotlib.pyplot as plt
import numpy as np
from keras_segmentation.models.unet import vgg_unet
from keras.callbacks import Callback
from keras_segmentation.predict import model_from_checkpoint_path
from keras import backend as K


# tracking with wandb
wandb.init(
    name = "SCDD_20211104_predict_from_checkpoint_05_inference_mode",
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
checkpoint_path ="/SCDD-image-segmentation-keras/checkpoint/SCDD_20211104_vgg_unet/SCDD_vgg_unet_epoch_05.h5"

# Paths to save prediction
prediction_output_dir = "/SCDD-image-segmentation-keras/share/predictions_SCDD_20211104"
if not os.path.exists(prediction_output_dir):
    os.makedirs(prediction_output_dir)

# Load the model
if os.path.exists(checkpoint_path):
    model = vgg_unet(n_classes=wandb.config.n_classes ,  input_height=wandb.config.input_height, input_width=wandb.config.input_width)
    model.trainable = False
    K.set_learning_phase(0)  # Set to inference mode
    model.load_weights(checkpoint_path)
else:
    raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

# Custom WandB callback to log loss and accuracy after each batch/epoch
class WandbCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        """Log loss and accuracy after each epoch."""
        wandb.log({
            "epoch": epoch + 1,
            "loss": logs.get('loss'),
            "accuracy": logs.get('accuracy')
        })

    def on_batch_end(self, batch, logs=None):
        """Log loss and accuracy after each batch."""
        wandb.log({
            "batch": batch + 1,
            "batch_loss": logs.get('loss'),
            "batch_accuracy": logs.get('accuracy')
        })


# Get test image file names
test_images = os.listdir(test_image_path)

# Loop over all test images, make predictions and save them
for img_name in test_images:
    img_path = os.path.join(test_image_path, img_name)
    output_file = os.path.join(prediction_output_dir, f"pred_{img_name}")
    
    # Predict segmentation
    out = model.predict_segmentation(
        inp=img_path,
        out_fname=output_file
    )
    
    # Log prediction to WandB
    wandb.log({"predictions": [wandb.Image(output_file, caption=f"Prediction: {img_name}")]})
    plt.imshow(out)



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