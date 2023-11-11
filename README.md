# Skin Cancer Detection

## Project Description

This project is a comprehensive guide to building and training a deep learning model for image classification using TensorFlow and Keras. The code provided in this repository includes a Python script (`main.py`) and a utility module (`utilities.py`) to perform the following tasks:

1. **Data Loading and Preprocessing**: The project starts by loading and preprocessing image data for a binary classification task. The dataset is assumed to be structured with images in different subdirectories.You can download the image dataset from [here](https://drive.google.com/drive/folders/1BAC4wJNksepPk3fquLF-DiWdS4nnIE1G?usp=share_link).

2. **Data Visualization**: The `Visualization` class in `utilities.py` offers various functions to visualize the data, such as displaying class distribution, sample images for each class, and plotting AUC and loss graphs.

3. **Model Creation and Training**: The project demonstrates three different types of models: a custom CNN, transfer learning with a pre-trained ResNet50 model, and a custom residual network (ResNet-like) model. You can choose which model to use by uncommenting the corresponding lines in the `main.py` script. The `HandleModel` class in `utilities.py` is responsible for creating, compiling, and training these models.

4. **Data Augmentation**: You can apply data augmentation to the training dataset by uncommenting relevant lines in the `main.py` script. Data augmentation is a technique to artificially increase the size of the training dataset by applying random transformations to the images. The `ProcessData` class in `utilities.py` provides the necessary functions for data augmentation.

5. **Model Evaluation and Prediction**: After training the model, you can evaluate its performance by displaying metrics such as classification report and confusion matrix. Additionally, the model is used to make predictions on a test dataset, and the results are saved in a CSV file.

## Requirements

The project requires the following Python libraries:

- `numpy`
- `pandas`
- `matplotlib`
- `PIL` (Python Imaging Library)
- `tensorflow` (TensorFlow 2.x)
- `scikit-learn` (for classification report and confusion matrix)

## Usage

1. Clone this repository or download the provided files to your local machine.

2. Ensure that you have the required libraries installed. You can install them using `pip`:

   ```bash
   pip install numpy pandas matplotlib pillow tensorflow scikit-learn
