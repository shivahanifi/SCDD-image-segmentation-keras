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
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping
from keras_segmentation.predict import evaluate_and_plot_confusion_matrix
import keras.backend as K
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
physical_devices = tf.config.experimental.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

# Main path
main_path = os.path.join(custom_path, "share")

# Train images and annotations path
train_image_path = os.path.join(main_path, "dataset_20221008/el_images_train")
train_annotation_path = os.path.join(main_path, "dataset_20221008/el_masks_train")

# Validation images and annotations path
val_image_path = os.path.join(main_path, "dataset_20221008/el_images_val")
val_annotation_path = os.path.join(main_path, "dataset_20221008/el_masks_val")

# Test images and annotations path
test_image_path = os.path.join(main_path,"dataset_20221008/el_images_test")
test_annotation_dir = os.path.join(main_path, "dataset_20221008/el_masks_test")

# CSV file for classes
csv_path = os.path.join(main_path, "dataset_20221008/ListOfClassesAndColorCodes_20221008.csv")
df = pd.read_csv(csv_path)
colors = df[['Red', 'Green','Blue']].apply(lambda x: (x['Red'], x['Green'], x['Blue']), axis=1).tolist()
class_names = df['Desc'].tolist()
class_labels = df['Label'].tolist()
class_dict = {class_labels[i]: class_names[i] for i in range(len(class_labels))}

# tracking with wandb
wandb.init(
    name = "vgg-unet",
    project="scdd_segmentation_keras", 
    entity="ubix",
    config={
        "architecture": "vgg_unet",
        "dataset": "SCDD_20211104_augmented",
        "n_classes": 29,
        "input_height": 416,
        "input_width": 608,
        "epochs":30,
        "batch_size":8,
        #"steps_per_epoch":1,
        #"steps_per_epoch":len(os.listdir(train_image_path))//wandb.config.batch_size,
        #"val_steps_per_epoch":len(os.listdir(val_image_path))//wandb.config.val_batch_size,
        "colors":colors,
        "labels_Desc":class_names,
        "labels": class_labels,
        "class_dict": class_dict,
        "val_batch_size":16,
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

focus_classes = [0, 1, 4, 14, 15]
class_weights = [
    0.15,  # "bckgnd"
    0.25,  # "sp multi"
    0.25,  # "sp mono"
    0.27,  # "sp dogbone"
    0.30,  # "ribbons"
    0.25,  # "border"
    0.27,  # "text"
    0.20,  # "padding"
    0.27,  # "clamp"
    0.25,  # "busbars"
    0.27,  # "crack rbn edge"
    0.40,  # "inactive"
    0.27,  # "rings"
    0.25,  # "material"
    0.45,  # "crack"
    0.35,  # "gridline"
    0.25,  # "splice"
    0.27,  # "dead cell"
    0.25,  # "corrosion"
    0.27,  # "belt mark"
    0.25,  # "edge dark"
    0.25,  # "frame edge"
    0.27,  # "jbox"
    0.27,  # "sp mono halfcut"
    0.25,  # "scuff"
    0.25,  # "corrosion cell"
    0.25,  # "brightening"
    0.25,  # "star"
]
wandb.config.class_weights = class_weights

class PrecisionRecallMatrixCallback(Callback):
    def __init__(self, val_data, class_names, focus_classes):
        """
        Custom callback to compute precision and recall matrices for selected classes.
        Args:
            val_data: Tuple of validation data (val_images, val_labels)
            class_names: List of all class names
            focus_classes: List of class indices to focus on
        """
        super().__init__()
        self.val_data = val_data
        self.class_names = class_names
        self.focus_classes = focus_classes  # Indices of classes to include

    def on_epoch_end(self, epoch, logs=None):
        val_images, val_labels = self.val_data
        val_pred = np.argmax(self.model.predict(val_images), axis=-1)
        val_true = np.argmax(val_labels, axis=-1)

        # Filter predictions and labels for focus classes
        mask = np.isin(val_true.flatten(), self.focus_classes)
        filtered_true = val_true.flatten()[mask]
        filtered_pred = val_pred.flatten()[mask]

        # Remap the class indices to the range [0, len(focus_classes) - 1]
        mapping = {class_idx: i for i, class_idx in enumerate(self.focus_classes)}
        remapped_true = np.array([mapping[cls] for cls in filtered_true])
        remapped_pred = np.array([mapping[cls] for cls in filtered_pred])

        # Compute confusion matrix
        cm = confusion_matrix(remapped_true, remapped_pred)
        cm_recall = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_precision = cm.astype('float') / cm.sum(axis=0)[np.newaxis, :]
        
        # Plot Confusion Matrix
        focus_class_names = [self.class_names[i] for i in self.focus_classes]
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm',
                    xticklabels=focus_class_names, yticklabels=focus_class_names)
        plt.title(f'Confusion Matrix (Epoch {epoch + 1})')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        confusion_img_path = f"confusion_matrix_epoch_{epoch + 1}.png"
        plt.savefig(confusion_img_path)
        plt.close()

        # Log Confusion Matrix to WandB
        wandb.log({f"Confusion Matrix (Epoch {epoch + 1})": wandb.Image(confusion_img_path)})


        # Plot Precision Matrix
        focus_class_names = [self.class_names[i] for i in self.focus_classes]
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_precision, annot=True, fmt='.2f', cmap='viridis',
                    xticklabels=focus_class_names, yticklabels=focus_class_names)
        plt.title(f'Precision Matrix (Epoch {epoch + 1})')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        precision_img_path = f"precision_matrix_epoch_{epoch + 1}.png"
        plt.savefig(precision_img_path)
        plt.close()

        # Log Precision Matrix to WandB
        wandb.log({f"Precision Matrix (Epoch {epoch + 1})": wandb.Image(precision_img_path)})

        # Plot Recall Matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_recall, annot=True, fmt='.2f', cmap='viridis',
                    xticklabels=focus_class_names, yticklabels=focus_class_names)
        plt.title(f'Recall Matrix (Epoch {epoch + 1})')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        recall_img_path = f"recall_matrix_epoch_{epoch + 1}.png"
        plt.savefig(recall_img_path)
        plt.close()

        # Log Recall Matrix to WandB
        wandb.log({f"Recall Matrix (Epoch {epoch + 1})": wandb.Image(recall_img_path)})


# #precision_recall_callback = PrecisionRecallMatrixCallback(
#     val_data=(test_image_path, test_annotation_dir),
#     class_names=class_names,
#     focus_classes=focus_classes,
# )

# Define the model 
model = vgg_unet(n_classes=wandb.config.n_classes ,  input_height=wandb.config.input_height, input_width=wandb.config.input_width)

# # Define the loss function with the computed class weights
# wcce = model.WeightedCategoricalCrossentropy(weights)
# custom_loss = wcce(targets,predictions)
# # Re-compile the model with the custom loss function
# model.compile(optimizer='adam', loss=custom_loss, metrics=['accuracy'])


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
            "accuracy": logs.get('accuracy'),
            "val_loss": logs.get('val_loss'),  
            "val_accuracy": logs.get('val_accuracy'),
        })

    def on_batch_end(self, batch, logs=None):
        """Log loss and accuracy after each batch."""
        wandb.log({
            "batch": batch + 1,
            "batch_loss": logs.get('loss'),
            "batch_accuracy": logs.get('accuracy')
        })
    
    def on_val_batch_end(self, batch, logs=None):
        """Log validation loss and accuracy after each validation batch."""
        logs = logs or {}
        wandb.log({
            "val_batch": batch + 1,
            "val_batch_loss": logs.get('loss'),  # Validation batch loss
            "val_batch_accuracy": logs.get('accuracy')  # Validation batch accuracy
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

# Early stopping Callback
early_stopping = EarlyStopping(
    monitor='val_loss',        
    patience=3,                
    restore_best_weights=True 
)

# Train the model with the custom wandb callback
model.train(
    train_images=train_image_path,
    train_annotations=train_annotation_path,
    checkpoints_path=checkpoint_path,
    epochs=wandb.config.epochs,
    batch_size = len(os.listdir(train_image_path))//wandb.config.batch_size,
    validate=True,
    val_images=val_image_path,
    val_annotations=val_annotation_path,
    val_batch_size=len(os.listdir(val_image_path))//wandb.config.val_batch_size,
    steps_per_epoch=len(os.listdir(val_image_path))//wandb.config.val_batch_size,
    val_steps_per_epoch=len(os.listdir(val_image_path)),
    do_augment=True,
    augmentation_name="aug_all",
    callbacks=[WandbCallback(), checkpoint_callback, early_stopping],
    class_weights=class_weights,
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
    gt_mask_dir=test_annotation_dir,
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

cm_evaluation = evaluate_and_plot_confusion_matrix(model=model, inp_images_dir= test_image_path , annotations_dir= test_annotation_dir, class_names=class_names,)
print(cm_evaluation)

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
