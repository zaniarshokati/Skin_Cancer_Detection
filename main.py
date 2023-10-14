import numpy as np
import pandas as pd
from utilities import Visualization, ImageProcessing, CreateModel

from glob import glob
from sklearn.model_selection import train_test_split
import tensorflow as tf

AUTO = tf.data.experimental.AUTOTUNE
import warnings

warnings.filterwarnings("ignore")


def main():
    # Initialize the visualization object
    visualize = Visualization()

    # Load and preprocess data
    df = load_data()

    # Show class distribution and class sample (optional)
    # visualize.show_class_distribution(df)
    # visualize.show_class_sample(df)

    # Process data and create training and validation datasets
    train_ds, val_ds = process_data(df)

    # Create and compile the model
    model_creator = CreateModel()
    model = model_creator.create_model()
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["AUC"])
    print(model.summary())

    # Train the model and save the training history
    history = model.fit(train_ds, validation_data=val_ds, epochs=15, verbose=1)

    # Display loss and AUC graphs
    visualize.show_loss_val(history)
    visualize.show_AUC_val(history)


def load_data():
    # Get a list of image file paths
    images = glob("dataset_2/train/*/*.jpg")
    # Replace backslash with forward slash to avoid unexpected errors
    images = [path.replace("\\", "/") for path in images]

    # Create a DataFrame with file paths, labels, and binary labels
    df = pd.DataFrame({"filepath": images})
    df["label"] = df["filepath"].str.split("/", expand=True)[2]
    df["label_bin"] = np.where(df["label"].values == "malignant", 1, 0)

    return df


def process_data(df):
    # Split data into features and targets
    features = df["filepath"]
    target = df["label_bin"]

    # Split the data into training and validation sets
    X_train, X_val, Y_train, Y_val = train_test_split(
        features, target, test_size=0.15, random_state=10
    )

    # Initialize the image processing object
    decoder = ImageProcessing()

    # Create training dataset
    train_ds = (
        tf.data.Dataset.from_tensor_slices((X_train, Y_train))
        .map(decoder.decode_image, num_parallel_calls=AUTO)
        .batch(32)
        .prefetch(AUTO)
    )

    # Create validation dataset
    val_ds = (
        tf.data.Dataset.from_tensor_slices((X_val, Y_val))
        .map(decoder.decode_image, num_parallel_calls=AUTO)
        .batch(32)
        .prefetch(AUTO)
    )

    return train_ds, val_ds


if __name__ == "__main__":
    main()
