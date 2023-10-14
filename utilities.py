import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pandas as pd
import tensorflow as tf
from keras import layers
from tensorflow.keras import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.applications.efficientnet import EfficientNetB7
from tensorflow.keras.applications.resnet50 import ResNet50


# Visualization class for displaying data-related information
class Visualization:
    def show_class_distribution(self, df):
        # Show a pie chart of class distribution
        x = df["label"].value_counts()
        plt.pie(x.values, labels=x.index, autopct="%1.1f%%")
        plt.show()

    def show_class_sample(self, df):
        # Show sample images for each class
        for cat in df["label"].unique():
            temp = df[df["label"] == cat]
            index_list = temp.index
            fig, ax = plt.subplots(1, 4, figsize=(15, 5))
            fig.suptitle(f"Images for {cat} category . . . .", fontsize=20)
            for i in range(4):
                index = np.random.choice(index_list)
                data = df.iloc[index]
                image_path = data[0]
                img = np.array(Image.open(image_path))
                ax[i].imshow(img)
            plt.tight_layout()
            plt.show()

    def show_AUC_val(self, history):
        # Show AUC and Validation AUC plot
        hist_df = pd.DataFrame(history.history)
        hist_df[["auc", "val_auc"]].plot()
        plt.title("AUC v/s Validation AUC")
        plt.legend()
        plt.show()

    def show_loss_val(self, history):
        # Show loss and Validation Loss plot
        hist_df = pd.DataFrame(history.history)
        hist_df[["loss", "val_loss"]].plot()
        plt.title("Loss v/s Validation Loss")
        plt.legend()
        plt.show()


# ImageProcessing class for image decoding and preprocessing
class ImageProcessing:
    def decode_image(self, filepath, label=None):
        # Decode and preprocess an image
        img = tf.io.read_file(filepath)
        img = tf.image.decode_jpeg(img)
        img = tf.image.resize(img, [224, 224])
        img = tf.cast(img, tf.float32) / 255.0

        if label is None:
            return img
        return img, label


# CreateModel class for creating a neural network model
class CreateModel:
    def create_model(self):
        # Create a neural network model based on ResNet50
        base_model = ResNet50(
            weights="imagenet", include_top=False, input_shape=(224, 224, 3)
        )
        for layer in base_model.layers[5:]:
            layer.trainable = False

        inputs = layers.Input(shape=(224, 224, 3))
        x = base_model(inputs)
        x = layers.Flatten()(x)
        x = layers.Dense(512, activation="relu")(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(1, activation="sigmoid")(x)
        model = Model(inputs, outputs)

        return model
