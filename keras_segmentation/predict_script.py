import wandb
import matplotlib.pyplot as plt
from keras_segmentation.models.unet import vgg_unet
from keras.callbacks import Callback


# tracking with wandb
wandb.init(
    project="scdd_segmentation_keras", 
    entity="ubix",
    config={
        "architecture": "vgg_unet",
        "dataset": "example_dataset",
        "n_classes": 51,
        "input_height": 416,
        "input_width": 608,
        "epochs":5,
    })

# train images and annotations path
train_image_path = "/SCDD-image-segmentation-keras/test/example_dataset/images_prepped_train"
train_annotations_path = "/SCDD-image-segmentation-keras/test/example_dataset/annotations_prepped_train"

# test images and annotations path
test_image_path = "/SCDD-image-segmentation-keras/test/example_dataset/images_prepped_test"
test_annotation_dir ="/SCDD-image-segmentation-keras/test/example_dataset/annotations_prepped_test"

# Checkpoint path
checkpoint_path ="/SCDD-image-segmentation-keras/checkpoint/example_vgg_unet_1"

model = vgg_unet(n_classes=wandb.config.n_classes ,  input_height=wandb.config.input_height, input_width=wandb.config.input_width)

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

# Train the model with the custom callback
model.train(
    train_images=train_image_path,
    train_annotations=train_annotations_path,
    checkpoints_path=checkpoint_path,
    epochs=wandb.config.epochs,
    callbacks=[WandbCallback()]  # Add the custom callback here
)


out = model.predict_segmentation(
    inp="/SCDD-image-segmentation-keras/test/example_dataset/images_prepped_test/0016E5_07959.png",
    out_fname="/SCDD-image-segmentation-keras/share/example_out.png"
)
# Log the prediction
wandb.log({"predictions": [wandb.Image(out, caption="Predicted Segmentation")]})


plt.imshow(out)

# evaluating the model 
evaluation_result= model.evaluate_segmentation( inp_images_dir= test_image_path , annotations_dir= test_annotation_dir)

print(evaluation_result)

# Log evaluation results
if 'iou' in evaluation_result:
    wandb.log({"iou": evaluation_result['iou']})

# Finish the WandB run
wandb.finish()