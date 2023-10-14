import numpy as np
import pandas as pd

import utilities

from glob import glob

from sklearn.model_selection import train_test_split

import tensorflow as tf

AUTO = tf.data.experimental.AUTOTUNE
import warnings
warnings.filterwarnings('ignore')

visualize = utilities.Visualization()
images = glob('dataset_2/train/*/*.jpg')
#replace backslash with forward slash to avoid unexpected errors
images = [path.replace('\\', '/') for path in images]
df = pd.DataFrame({'filepath': images})
df['label'] = df['filepath'].str.split('/', expand=True)[2]
df['label_bin'] = np.where(df['label'].values == 'malignant', 1, 0)

# visualize.show_class_distribution(df)
# visualize.show_class_sample(df)


features = df['filepath']
target = df['label_bin']

X_train, X_val, Y_train, Y_val = train_test_split(features, target,
									test_size=0.15,
									random_state=10)

decoder= utilities.ImageProcessing()

train_ds = (
	tf.data.Dataset
	.from_tensor_slices((X_train, Y_train))
	.map(decoder.decode_image, num_parallel_calls=AUTO)
	.batch(32)
	.prefetch(AUTO)
)

val_ds = (
	tf.data.Dataset
	.from_tensor_slices((X_val, Y_val))
	.map(decoder.decode_image, num_parallel_calls=AUTO)
	.batch(32)
	.prefetch(AUTO)
)



model_creator = utilities.CreateModel()
model = model_creator.create_model()


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])
print(model.summary())


history = model.fit(train_ds,
					validation_data=val_ds,
					epochs=5,
					verbose=1)
visualize.show_loss_val(history)
visualize.show_AUC_val(history)