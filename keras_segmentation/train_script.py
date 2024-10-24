import wandb
import os
import io
import matplotlib.pyplot as plt
import numpy as np
from keras_segmentation.models.unet import vgg_unet
from keras.callbacks import Callback
from keras_segmentation.train import CheckpointsCallback
from contextlib import redirect_stdout
import keras.backend as K

# tracking with wandb
wandb.init(
    name = "train_SCDD_20211104_augmented",
    project="scdd_segmentation_keras", 
    entity="ubix",
    config={
        "architecture": "vgg_unet",
        "dataset": "SCDD_20211104_augmented",
        "n_classes": 24,
        "input_height": 416,
        "input_width": 608,
        "epochs":5,
    })

# Train images and annotations path
train_image_path = "/SCDD-image-segmentation-keras/share/SCDD_20211104/images_train_augmented"
train_annotations_path = "/SCDD-image-segmentation-keras/share/SCDD_20211104/masks_coded_train_augmented"

# Test images and annotations path
test_image_path = "/SCDD-image-segmentation-keras/share/SCDD_20211104/images_test"
test_annotation_dir ="/SCDD-image-segmentation-keras/share/SCDD_20211104/masks_coded_test"

# Checkpoint path
checkpoint_path ="/SCDD-image-segmentation-keras/checkpoint/SCDD_20211104_augmented_vgg_unet/"
if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)

# Paths to save prediction
prediction_output_dir = "/SCDD-image-segmentation-keras/share/predictions_SCDD_20211104_augmented"
if not os.path.exists(prediction_output_dir):
    os.makedirs(prediction_output_dir)
    

# Capture model summary
def get_model_summary(model):
    stream = io.StringIO()
    with redirect_stdout(stream):
        model.summary()
    return stream.getvalue()

# Define the model 
model = vgg_unet(n_classes=wandb.config.n_classes ,  input_height=wandb.config.input_height, input_width=wandb.config.input_width)

# Log model summary to wandb
model_summary = get_model_summary(model)
wandb.summary['model summary'] = model_summary


# Log total parameters
total_params = model.count_params()

# Calculate trainable and non-trainable parameters
trainable_params = np.sum([K.count_params(w) for w in model.trainable_weights])
non_trainable_params = np.sum([K.count_params(w) for w in model.non_trainable_weights])

# Log trainable and non-trainable parameters to WandB
# wandb.log({
#     "total_params": total_params,
#     "trainable_params": trainable_params,
#     "non_trainable_params": non_trainable_params
# })
wandb.config.total_params = total_params
wandb.config.trainable_params = trainable_params
wandb.config.non_trainable_params = non_trainable_params

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

# Initialize the CheckpointsCallback
checkpoint_callback = CheckpointsCallback(checkpoint_path)

# Train the model with the custom wandb callback
model.train(
    train_images=train_image_path,
    train_annotations=train_annotations_path,
    checkpoints_path=checkpoint_path,
    epochs=wandb.config.epochs,
    batch_size = 2,
    steps_per_epoch=len(os.listdir(train_image_path)) // 2,
    callbacks=[WandbCallback(), checkpoint_callback] 
)

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

# Log loss function and layer information
wandb.config.update({
    "loss_function": model.loss,
    "optimizer": model.optimizer.get_config() if model.optimizer else "Not Defined",
    "layers": [layer.name for layer in model.layers]
})

# evaluating the model 
evaluation_result= model.evaluate_segmentation( inp_images_dir= test_image_path , annotations_dir= test_annotation_dir)

wandb.save(os.path.join(checkpoint_path, "*"))

print(evaluation_result)

# Log evaluation results
wandb.log({"frequency_weighted_IU": evaluation_result['frequency_weighted_IU'], 
            "mean_IU": evaluation_result['mean_IU'], 
            "class_wise_IU": evaluation_result['class_wise_IU'], 
            })

# Finish the WandB run
wandb.finish()