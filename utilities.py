import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pandas as pd
import tensorflow as tf
from keras import layers
from glob import glob
import os
from sklearn.metrics import confusion_matrix
import seaborn as sns

from sklearn.model_selection import train_test_split
from tensorflow.keras import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.applications.efficientnet import EfficientNetB7
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator


AUTO = tf.data.experimental.AUTOTUNE


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

    def show_confusion_matrix(self,true_labels, predicted_labels):
        cm = confusion_matrix(true_labels, predicted_labels)
        # Create a heatmap of the confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.show()


# ImageProcessing class for image decoding and preprocessing
class ProcessData:

    def load_train_data(self,train_path):
        # Get a list of image file paths
        images = glob(train_path)
        # Replace backslash with forward slash to avoid unexpected errors
        # images = [path.replace("\\", "/") for path in images]

        # Create a DataFrame with file paths, labels, and binary labels
        df = pd.DataFrame({"filepath": images})
        df["label"] = df["filepath"].str.split("/", expand=True)[2]
        df["label_bin"] = np.where(df["label"].values == "malignant", 1, 0)
        df["label_bin"] = df["label_bin"].astype(int)

        return df
    
    def load_test_data(self,test_path):
        # Get a list of image file paths
        path_list = sorted(glob(test_path))
        # Replace backslash with forward slash to avoid unexpected errors
        # images = [path.replace("\\", "/") for path in images]
        file_names = [os.path.basename(path) for path in path_list]
        # Create a DataFrame with file paths, labels, and binary labels
        df = pd.DataFrame({"filename": path_list})
        directory_names = [os.path.dirname(path).split(os.path.sep)[-1] for path in path_list]
        df["label"] = directory_names
        df["label_bin"] = np.where(df["label"].values == "malignant", 1, 0)
        df.to_csv("test_data.csv", index=False)
        return df
    
    def decode_image(self, filepath, label=None):
        # Decode and preprocess an image
        img = tf.io.read_file(filepath)
        img = tf.image.decode_jpeg(img)
        img = tf.image.resize(img, [224, 224])
        img = tf.cast(img, tf.float32) / 255.0

        if label is None:
            return img
        return img, label
    
    def process_data(self, df):
        # Split data into features and targets
        features = df["filepath"]
        target = df["label_bin"].values  # Convert labels to a NumPy array

        # Split the data into training and validation sets
        X_train, X_val, Y_train, Y_val = train_test_split(
            features, target, test_size=0.15, random_state=10
        )

        # Create training dataset
        train_ds = (
            tf.data.Dataset.from_tensor_slices((X_train, Y_train))
            .map(self.decode_image, num_parallel_calls=AUTO)
            .batch(32)
            .prefetch(AUTO)
        )

        # Create validation dataset
        val_ds = (
            tf.data.Dataset.from_tensor_slices((X_val, Y_val))
            .map(self.decode_image, num_parallel_calls=AUTO)
            .batch(32)
            .prefetch(AUTO)
        )

        return train_ds, val_ds

    
    def process_test_data(self, df):
        features = df["filename"]  # Assuming "filename" is the column containing the file paths
        test_ds = (
            tf.data.Dataset.from_tensor_slices(features)
            .map(self.decode_image, num_parallel_calls=AUTO)
            .batch(32)
            .prefetch(AUTO)
        )
        print(test_ds)
        return test_ds

    def data_augmentation(self):
        data_gen = ImageDataGenerator(
            horizontal_flip=True,
            vertical_flip=True,
            rotation_range=10,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.1,
            zoom_range=0.2,
            brightness_range=(0.8, 1.2)
        )

        return data_gen
    
    def process_data_with_augmentation(self, df):

        features = df["filepath"]
        target = df["label_bin"].values

        X_train, X_val, Y_train, Y_val = train_test_split(
            features, target, test_size=0.15, random_state=10
        )

        train_ds = (
            tf.data.Dataset.from_tensor_slices((X_train, Y_train))
            .map(self.decode_image, num_parallel_calls=AUTO)
            .batch(32)
            .prefetch(AUTO)
        )

        val_ds = (
            tf.data.Dataset.from_tensor_slices((X_val, Y_val))
            .map(self.decode_image, num_parallel_calls=AUTO)
            .batch(32)
            .prefetch(AUTO)
        )

        data_gen = self.data_augmentation()
        augmented_train_ds = data_gen.flow(train_ds, batch_size=32)

        return augmented_train_ds, val_ds

# CreateModel class for creating a neural network model
class HandleModel:
    def residual_block(self, input_tensor, num_filters, stride=1):

        x = layers.Conv2D(num_filters, kernel_size=(3, 3), strides=stride, padding='same')(input_tensor)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv2D(num_filters, kernel_size=(3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        
        if stride != 1 or input_tensor.shape[-1] != num_filters:
            input_tensor = layers.Conv2D(num_filters, kernel_size=(1, 1), strides=stride, padding='same')(input_tensor)
        
        x = layers.Add()([input_tensor, x])
        x = layers.ReLU()(x)
    
        return x
    
    def residual_model(self,input_shape, num_blocks=3, num_filters=32):
        inputs = layers.Input(shape=input_shape)
        
        x = layers.Conv2D(num_filters, (7, 7), strides=2, padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        
        for _ in range(num_blocks):
            x = self.residual_block(x, num_filters)
        
        x = layers.GlobalAveragePooling2D()(x)
        outputs = layers.Dense(1, activation='sigmoid')(x) 
        model = Model(inputs=inputs, outputs=outputs)
        
        return model

    def create_model_cnn(self,input_shape):

        model = Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(128, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(1, activation='sigmoid'))

        return model
    
    def create_model_transfer_learning(self):
        # Create a neural network model based on ResNet50
        base_model = ResNet50(
            weights="imagenet", include_top=False, input_shape=(224, 224, 3)
        )
        for layer in base_model.layers[:45]:
            layer.trainable = False

        inputs = layers.Input(shape=(224, 224, 3))
        x = base_model(inputs)
        x = layers.Flatten()(x)
        x = layers.Dense(512, activation="relu")(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(1, activation="sigmoid")(x)
        model = Model(inputs, outputs)
        
        return model
    
    def compile_model(self, model):
        initial_learning_rate = 0.001
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate, decay_steps=10000, decay_rate=0.9
        )
        optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=lr_schedule)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['AUC'])
        return model    
    
    def train_model(self, model, train_ds, val_ds):
        early = EarlyStopping(monitor='val_loss', patience=10, min_delta=0.001, restore_best_weights=True)
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=50,
            callbacks=[early]
        )
        model.save("Model.h5")
        return history

