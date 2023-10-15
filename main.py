import numpy as np
import pandas as pd
from utilities import Visualization, ProcessData, HandleModel
from sklearn.metrics import classification_report

from sklearn.model_selection import train_test_split
import tensorflow as tf

AUTO = tf.data.experimental.AUTOTUNE
import warnings

warnings.filterwarnings("ignore")
class Application:
	def __init__(self) -> None:
		self.visualize = Visualization()
		self.data_processor = ProcessData()
		self.model_handler = HandleModel()

	def main(self):
		# Load and preprocess data
		train_path = "dataset_2/train/*/*.jpg"
		test_path = "dataset_2/test/*/*.jpg"
		df_train = self.data_processor.load_train_data(train_path)
		df_test = self.data_processor.load_test_data(test_path)
		# Process the test data
		test_ds = self.data_processor.process_test_data(df_test)
		# Show class distribution and class sample (optional)
		# visualize.show_class_distribution(df)
		# visualize.show_class_sample(df)
		
		# Process data and create training and validation datasets
		train_ds, val_ds = self.data_processor.process_data(df_train)



		# Create and compile the model
		# CNN
		input_shape = (224, 224, 3) 
		model = self.model_handler.create_model_cnn(input_shape)
		model = self.model_handler.compile_model(model)
		# Transfer Learning 
		# model = self.model_handler.create_model_transfer_learning()
		# model = self.model_handler.compile_model(model)
										   
		# RenNet Model
		# input_shape = (224, 224, 3) 
		# num_classes = 2  # Change this according to the number of classes in your dataset
		# model = self.model_handler.residual_model(input_shape)
		# model = self.model_handler.compile_model(model)
		print(model.summary())

		# Train the model and save the training history
		# history = model.fit(train_ds, validation_data=val_ds, epochs=5, verbose=1)
		history = self.model_handler.train_model(model, train_ds, val_ds)
		# Display loss and AUC graphs
		self.visualize.show_loss_val(history)
		self.visualize.show_AUC_val(history)
		
		# Load the trained model
		model = tf.keras.models.load_model("Model.h5")

		# Make predictions on the test data
		predictions = model.predict(test_ds)

		df_test["predictions"] = predictions  # Add a new column for predictions
		df_test.to_csv("test_predictions.csv", index=False)
		y_true = df_test["label_bin"]
		y_pred = (predictions > 0.5).astype(int)  # Convert probabilities to binary predictions
		report = classification_report(y_true, y_pred)
		print(report)

if __name__ == "__main__":
    app = Application()
    app.main()
