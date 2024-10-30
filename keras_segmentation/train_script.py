import wandb
import os
import io
import csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from keras_segmentation.models.unet import vgg_unet
from keras.callbacks import Callback, ModelCheckpoint
from keras_segmentation.train import CheckpointsCallback
from contextlib import redirect_stdout
import keras.backend as K

# Main path
main_path = "/SCDD-image-segmentation-keras/share/"

# Train images and annotations path
train_image_path = os.path.join(main_path, "SCDD_20211104/images_train_augmented")
train_annotations_path = os.path.join(main_path, "SCDD_20211104/masks_coded_train_augmented")

# Test images and annotations path
test_image_path = os.path.join(main_path,"SCDD_20211104/images_test")
test_annotation_dir = os.path.join(main_path, "SCDD_20211104/masks_coded_test")

# CSV file for classes
csv_path = os.path.join(main_path, "SCDD_20211104/ListOfClassesAndColorCodes_20211104.csv")
df = pd.read_csv(csv_path)
colors = df[['Red', 'Green','Blue']].apply(lambda x: (x['Red'], x['Green'], x['Blue']), axis=1).tolist()
class_names = df['Desc'].tolist()

# tracking with wandb
run = wandb.init(
    name = "train_SCDD_20211104_augmented_e1_speFull",
    project="scdd_segmentation_keras", 
    entity="ubix",
    config={
        "architecture": "vgg_unet",
        "dataset": "SCDD_20211104_augmented",
        "n_classes": 24,
        "input_height": 416,
        "input_width": 608,
        "epochs":1,
        "batch_size":2,
        "steps_per_epoch":1,
        "colors":colors,
        "labels_Desc":class_names,
    })

# Checkpoint path
checkpoint_path = os.path.join(main_path, wandb.run.name, "checkpoint/")
if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)

# Paths to save prediction
prediction_output_dir = os.path.join(main_path , wandb.run.name, "predictions/")
if not os.path.exists(prediction_output_dir):
    os.makedirs(prediction_output_dir)
print(prediction_output_dir)  

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
#checkpoint_callback = CheckpointsCallback(checkpoint_path)

# Create ModelCheckpoint callback to save only the best model based on validation metric
checkpoint_callback = ModelCheckpoint(
    checkpoint_path,
    monitor="accuracy", 
    save_best_only=True,
    save_weights_only=False,  # Save entire model, not just weights
    verbose=1
)

# Train the model with the custom wandb callback
model.train(
    train_images=train_image_path,
    train_annotations=train_annotations_path,
    checkpoints_path=checkpoint_path,
    epochs=wandb.config.epochs,
    batch_size = wandb.config.batch_size,
    steps_per_epoch=wandb.config.steps_per_epoch,
    callbacks=[WandbCallback(), checkpoint_callback] 
)


artifact = wandb.Artifact(name = wandb.run.name, type = "model")
# List files in the checkpoint directory and add only files to the artifact
ckp_files = os.listdir(checkpoint_path)
for ckp in ckp_files:
    full_path = os.path.join(checkpoint_path, ckp)
    artifact.add_file(local_path=full_path, name=ckp)
artifact.save()
wandb.save(os.path.join(checkpoint_path, "*"))

# Predict segmentation
predictions = model.predict_multiple(
    inp_dir=test_image_path,
    out_dir=prediction_output_dir,
    class_names=class_names,
    show_legends=True,
    colors=colors,
)

all_images = {}
for out_frame in os.listdir(prediction_output_dir):
    image_name = os.path.splitext(out_frame)[0]
    all_images[image_name] = wandb.Image(os.path.join(prediction_output_dir, out_frame), caption=f"Prediction for {out_frame}")
wandb.log({"predictions": all_images})


# Log loss function and layer information
wandb.config.update({
    "loss_function": model.loss,
    "optimizer": model.optimizer.get_config() if model.optimizer else "Not Defined",
    "layers": [layer.name for layer in model.layers]
})

# evaluating the model 
evaluation_result= model.evaluate_segmentation( inp_images_dir= test_image_path , annotations_dir= test_annotation_dir)
print(evaluation_result)


# Prepare class-wise IoU for logging
class_wise_IU = evaluation_result['class_wise_IU']
class_iou_dict = {f"class_{i}:{class_names[i]}_IU": iou for i, iou in enumerate(class_wise_IU)}

# Create a wandb.Table for class-wise IoU logging
class_wise_IU_table = wandb.Table(columns=["Class Name", "Class Index", "IoU", "Run Name"])
run_name = wandb.run.name
for i, iou in enumerate(class_wise_IU):
    class_wise_IU_table.add_data(class_names[i], i, iou, run_name)
wandb.log({"class_wise_IU_table": class_wise_IU_table})

# Log evaluation results
wandb.log({"frequency_weighted_IU": evaluation_result['frequency_weighted_IU'], 
            "mean_IU": evaluation_result['mean_IU'], 
            "class_wise_IU": class_iou_dict, 
            "run_name": run_name,
            })

# Finish the WandB run
wandb.finish()
