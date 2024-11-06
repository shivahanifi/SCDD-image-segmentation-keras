import sys
# Add your custom path to sys.path at the beginning to prioritize it
custom_path = "/SCDD-image-segmentation-keras"
if custom_path not in sys.path:
    sys.path.insert(0, custom_path)    
import wandb
import os
import pandas as pd
import numpy as np
from keras_segmentation.models.unet import vgg_unet
from keras.callbacks import Callback, ModelCheckpoint
import keras.backend as K


# Main path
main_path = os.path.join(custom_path, "share")

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
class_labels = df['Label'].tolist()
class_dict = {class_labels[i]: class_names[i] for i in range(len(class_labels))}

# tracking with wandb
wandb.init(
    name = "train_SCDD_20211104_overlay_log_test",
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
        "steps_per_epoch":5,
        "colors":colors,
        "labels_Desc":class_names,
        "labels": class_labels,
        "class_dict": class_dict,
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


# class weight assignment - set C
weights = {
    0 : 0.2,  # "bckgnd" 
    1: 1.0,  # "sp multi" 
    2: 1.0,  # "sp mono" 
    3: 1.0,  # "sp dogbone" 
    4: 3.0,  # "ribbons"
    5: 1.0,  # "border" 
    6: 1.0,  # "text" 
    7: 1.0,  # "padding" 
    8: 1.0,  # "clamp" 
    9: 1.0,  # "busbars" 
    10: 1.0,  # "crack rbn edge" 
    11: 10.0, # "inactive" 
    12: 1.0,  # "rings" 
    13: 1.0,  # "material" 
    14: 20.0, # "crack" 
    15: 10.0, # "gridline" 
    16: 1.0,  # "splice" 
    17: 1.0,  # "dead cell"
    18: 1.0,  # "corrosion" 
    19: 1.0,  # "belt mark" 
    20: 1.0,  # "edge dark" 
    21: 1.0,  # "frame edge" 
    22: 1.0,  # "jbox" 
    23: 1.0,  # "meas artifact"
}

# Define weighted categorical cross-entropy loss
def weighted_categorical_crossentropy(weights):
    def loss(y_true, y_pred):
        y_true = K.one_hot(K.cast(K.flatten(y_true), 'int32'), num_classes=len(weights))
        y_pred = K.flatten(y_pred)
        # Log weights applied to the loss function
        print("Class Weights Applied in Loss Function:", weights)
        wandb.log({"actual_class_weights_used": weights})
        # Calculate weighted loss
        loss = K.categorical_crossentropy(y_true, y_pred)
        loss = K.sum(loss * K.constant(weights))
        return loss
    return loss


# Define the model 
model = vgg_unet(n_classes=wandb.config.n_classes ,  input_height=wandb.config.input_height, input_width=wandb.config.input_width)

# Define the loss function with the computed class weights
custom_loss = weighted_categorical_crossentropy(weights)
# Re-compile the model with the custom loss function
model.compile(optimizer='adam', loss=custom_loss, metrics=['accuracy'])


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

# Create ModelCheckpoint callback to save only the best model based on validation metric
checkpoint_callback = ModelCheckpoint(
    checkpoint_path,
    monitor="accuracy", 
    save_best_only=True,
    save_weights_only=False,  # Save entire model, not just weights
    save_freq="epoch", # Save model at the end of the epoch if it is the best
    verbose=1,
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

# After training, save the best model as an .h5 file
best_model_h5_path = os.path.join(checkpoint_path, f"{wandb.run.name}_best_model.h5")
model.save(best_model_h5_path)
artifact = wandb.Artifact(name=wandb.run.name, type="model")
artifact.add_dir(checkpoint_path)
wandb.log_artifact(artifact)

# Predict segmentation
predictions = model.predict_multiple(
    inp_dir=test_image_path,
    out_dir=prediction_output_dir,
    class_names=class_names,
    show_legends=True,
    colors=colors,
    class_dict=class_dict,
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
