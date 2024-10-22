import wandb
import os
import matplotlib.pyplot as plt
import numpy as np
from keras_segmentation.models.unet import vgg_unet
from keras.callbacks import Callback
from keras_segmentation.predict import model_from_checkpoint_path


# tracking with wandb
wandb.init(
    name = "SCDD_20211104_predict_from_checkpoint_h5",
    project="scdd_segmentation_keras", 
    entity="ubix",
    config={
        "architecture": "vgg_unet",
        "dataset": "example_dataset",
        "n_classes": 24,
        "input_height": 416,
        "input_width": 608,
        "epochs":1,
    })

# test images and annotations path
test_image_path = "/SCDD-image-segmentation-keras/share/SCDD_20211104/images_test"
test_annotation_dir ="/SCDD-image-segmentation-keras/share/SCDD_20211104/masks_coded_test"

# Checkpoint path
checkpoint_path ="/SCDD-image-segmentation-keras/checkpoint/SCDD_20211104_vgg_unet/SCDD_vgg_unet_epoch_01.h5"

# Paths to save prediction
prediction_output_dir = "/SCDD-image-segmentation-keras/share/predictions_SCDD_20211104"
if not os.path.exists(prediction_output_dir):
    os.makedirs(prediction_output_dir)

# Verify checkpoint exists
if os.path.exists(checkpoint_path):
    # Create and compile your model
    model = vgg_unet(n_classes=24, input_height=416, input_width=608)
    # After training, save the model
    model.save(checkpoint_path)
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