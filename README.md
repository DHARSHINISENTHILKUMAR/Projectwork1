## ALZHEIMER'S DISEASE DETECTION USING DEEP LEARNING
Alzheimer's disease is a progressive neurodegenerative disorder that primarily affects memory and cognitive functions. Detecting Alzheimer’s early is essential for providing timely treatment and improving the quality of life for patients. In recent years, deep learning has become a powerful tool for Alzheimer’s disease detection due to its ability to analyze complex data patterns within brain scans, genetic data, and other medical records. Models like Convolutional Neural Networks (CNNs) and Recurrent Neural Networks (RNNs) are commonly used in this domain to process imaging data from MRI or PET scans, identifying specific biomarkers or abnormalities that indicate the presence of Alzheimer’s.Deep learning models are trained on large datasets, including images of brains with and without Alzheimer's, allowing the algorithm to learn subtle differences that might be challenging for human experts to spot. These models have shown promising results in identifying early stages of Alzheimer’s, sometimes even before symptoms manifest. Additionally, such models can provide diagnostic support to healthcare professionals, potentially reducing the time and costs associated with Alzheimer’s testing and improving diagnostic accuracy. Through continuous advancements, deep learning holds the potential to revolutionize early diagnosis, paving the way for more effective treatment interventions for those at risk of developing Alzheimer’s disease.

## Features
Image Preprocessing: Clean and enhance MRI or PET scans.
Data Segmentation: Focus on key brain areas like the hippocampus.
Deep Learning Model: Use CNNs to detect Alzheimer’s patterns.
Feature Extraction: Identify important brain features for analysis.
Classification: Categorize images by Alzheimer’s stage or status.
Performance Metrics: Measure accuracy, recall, and precision.
Explainability: Visualize model focus areas with heat maps.
User Interface: Create an upload tool for easy predictions.
Cloud Integration: Enable faster, scalable processing.
Report Generation: Provide summary reports for clinicians.

## Requirements
Hardware: A computer with a good GPU to handle deep learning tasks efficiently.
Software: Python, deep learning libraries like TensorFlow or PyTorch, and data preprocessing tools.
Dataset: Access to brain MRI or PET scan datasets for Alzheimer’s, such as ADNI.
Cloud Storage (optional): For storing large datasets and computational power if needed.

## System Architecture
![image](https://github.com/user-attachments/assets/cb6a17e2-005a-40d4-bf0b-428905d79b10)
## Methodology:

## Program:
### Import packages:
```
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import seaborn as sns

import math

import os

import warnings

warnings.filterwarnings('ignore')

from sklearn.utils.class_weight import compute_class_weight

from sklearn.metrics import classification_report, confusion_matrix

import keras

from tensorflow import keras

from keras import Sequential

from keras import layers

import tensorflow as tf

from tensorflow.keras.preprocessing import image_dataset_from_directory

from tensorflow.keras import Sequential

from tensorflow.keras.utils import to_categorical

from tensorflow.keras.layers import Dense, Dropout, Activation, BatchNormalization, Flatten,

Conv2D, MaxPooling2D

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

plt.rcParams["figure.figsize"] = (10,6) plt.rcParams['figure.dpi'] = 300

colors = ["#B6EE56", "#D85F9C", "#EEA756", "#56EEE8"]
```

### Target value distribution:
```
if tf.test.gpu_device_name():

physical_devices = tf.config.experimental.list_physical_devices('GPU')

print('GPU active! -', physical_devices)

else:

print('GPU not active!')

except Exception as e:

print('An error occurred while checking the GPU:', e)

Generate TF Dataset

data = tf.keras.utils.image_dataset_from_directory(PATH,

batch_size = 32,

image_size=(128, 128),

shuffle=True,

seed=42,)

class_names = data.class_names

MRI Samples for Each Class

def sample_bringer(path, target, num_samples=5):

class_path = os.path.join(path, target)

image_files = [image for image in os.listdir(class_path) if image.endswith('.jpg')]

fig, ax = plt.subplots(1, num_samples, facecolor="gray")

fig.suptitle(f'{target} Brain MRI Samples', color="yellow",fontsize=16, fontweight='bold',

y=0.75)

for i in range(num_samples):

image_path = os.path.join(class_path, image_files[i])

img = mpimg.imread(image_path)

ax[i].imshow(img)

ax[i].axis('off')

ax[i].set_title(f'Sample {i+1}', color="aqua")

plt.tight_layout()

for target in class_names:

sample_bringer(PATH, target=target)

alz_dict = {index: img for index, img in enumerate(data.class_names)}
```
### Class process:
```
def init(self, data):

self.data = data.map(lambda x, y: (x/255, y))

def create_new_batch(self):

self.batch = self.data.as_numpy_iterator().next()

text = "Min and max pixel values in the batch ->"

print(text, self.batch[0].min(), "&", self.batch[0].max())

def show_batch_images(self, number_of_images=5):

fig, ax = plt.subplots(ncols=number_of_images, figsize=(20,20), facecolor="gray")

fig.suptitle("Brain MRI (Alzheimer) Samples in the Batch",

color="yellow",fontsize=18, fontweight='bold', y=0.6)

for idx, img in enumerate(self.batch[0][:number_of_images]):

ax[idx].imshow(img)

class_no = self.batch[1][idx]

ax[idx].set_title(alz_dict[class_no], color="aqua")

ax[idx].set_xticklabels([])

ax[idx].set_yticklabels([])

def train_test_val_split(self, train_size, val_size, test_size):

train = int(len(self.data)*train_size)

test = int(len(self.data)*test_size)

val = int(len(self.data)*val_size)

train_data = self.data.take(train)

val_data = self.data.skip(train).take(val)

test_data = self.data.skip(train+val).take(test)

return train_data, val_data, test_data

process = Process(data)

process.create_new_batch()

process.show_batch_images(number_of_images=5)

train_data, val_data, test_data= process.train_test_val_split(train_size=0.8, val_size=0.1,

test_size=0.1)

y_train = tf.concat(list(map(lambda x: x[1], train_data)), axis=0)

class_weight = compute_class_weight('balanced',classes=np.unique(y_train),

y=y_train.numpy())

class_weights = dict(zip(np.unique(y_train), class_weight))
```
### Model Building(CNN & RNN):
```
from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, SimpleRNN,
```
### Time distributed:
```
def build_model():

model = Sequential()

model.add(Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), activation="relu",

kernel_initializer='he_normal',

input_shape=(128, 128, 3)))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), activation="relu",

kernel_initializer='he_normal'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), activation="relu",

kernel_initializer='he_normal'))

model.add(MaxPooling2D(pool_size=(2, 2)))
```
### Add TimeDistributed layer to make the output 3D:
```
model.add(TimeDistributed(Flatten()))
```
### Add Simple RNN layer:
```
model.add(SimpleRNN(units=64, activation='relu'))

model.add(Dense(128, activation="relu", kernel_initializer='he_normal'))

model.add(Dense(64, activation="relu"))

model.add(Dense(4, activation="softmax"))

model.compile(optimizer='adam', loss="sparse_categorical_crossentropy",

metrics=['accuracy'])

model.summary()

return model

model = build_model()

def checkpoint_callback():

checkpoint_filepath = '/tmp/checkpoint'

model_checkpoint_callback= ModelCheckpoint(filepath=checkpoint_filepath,

save_weights_only=False,

frequency='epoch', monitor='val_accuracy',

save_best_only=True,

verbose=1)

return model_checkpoint_callback

def early_stopping(patience):

es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience,

verbose=1)

return es_callback

EPOCHS = 5

checkpoint_callback = checkpoint_callback()

early_stopping = early_stopping(patience=5)

callbacks = [checkpoint_callback, early_stopping]

Loss and Accuracy
```
### Save the trained model:
```
model.save("your_model_name.h5")

fig, ax = plt.subplots(1, 2, figsize=(12,6), facecolor="khaki")

ax[0].set_facecolor('palegoldenrod')

ax[0].set_title('Loss', fontweight="bold")

ax[0].set_xlabel("Epoch", size=14)

ax[0].plot(history.epoch, history.history["loss"], label="Train Loss", color="navy")

ax[0].plot(history.epoch, history.history["val_loss"], label="Validation Loss",

color="crimson", linestyle="dashed")

ax[0].legend()

ax[1].set_facecolor('palegoldenrod')

ax[1].set_title('Accuracy', fontweight="bold")

ax[1].set_xlabel("Epoch", size=14)

ax[1].plot(history.epoch, history.history["accuracy"], label="Train Acc.", color="navy") ax[1].plot(history.epoch, history.history["val_accuracy"], label="Validation Acc.",

color="crimson", linestyle="dashed")

ax[1].legend()

Evaluating Test Data

model.evaluate(test_data)

Classification Report

predictions = []

labels = []

for X, y in test_data.as_numpy_iterator():

y_pred = model.predict(X, verbose=0)

y_prediction = np.argmax(y_pred, axis=1)

predictions.extend(y_prediction)

labels.extend(y)

predictions = np.array(predictions)

labels = np.array(labels)

print(classification_report(labels, predictions, target_names=class_names))

Confusion Matrix

cm = confusion_matrix(labels, predictions)

cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)

cm_df

plt.figure(figsize=(10,6), dpi=300)

sns.heatmap(cm_df, annot=True, cmap="Greys", fmt=".1f")

plt.title("Confusion Matrix", fontweight="bold")

plt.xlabel("Predicted", fontweight="bold")

plt.ylabel("True", fontweight="bold") Alzheimer Probability of a Random MRI from Test Data

def random_mri_prob_bringer(image_number=0):

for images, _ in test_data.skip(5).take(1):

image = images[image_number]

pred = model.predict(tf.expand_dims(image, 0))[0]

probs = list(tf.nn.softmax(pred).numpy())

probs_dict = dict(zip(class_dist.keys(), probs))

keys = list(probs_dict.keys())

values = list(probs_dict.values())

fig, (ax1, ax2) = plt.subplots(1, 2, facecolor='black')

plt.subplots_adjust(wspace=0.4)

ax1.imshow(image)

ax1.set_title('Brain MRI', color="yellow", fontweight="bold", fontsize=16)

edges = ['left', 'bottom', 'right', 'top']

edge_color = "greenyellow"

edge_width = 3

for edge in edges:

ax1.spines[edge].set_linewidth(edge_width)

ax1.spines[edge].set_edgecolor(edge_color)

plt.gca().axes.yaxis.set_ticklabels([])

plt.gca().axes.xaxis.set_ticklabels([])

wedges, labels, autopct = ax2.pie(values, labels=keys, autopct='%1.1f%%',

shadow=True, startangle=90, colors=colors, textprops={'fontsize': 8,

"fontweight":"bold", "color":"white"}, wedgeprops=

{'edgecolor':'black'} , labeldistance=1.15) for autotext in autopct:

autotext.set_color('black')

ax2.set_title('Alzheimer Probabilities', color="yellow", fontweight="bold", fontsize=16)

rand_img_no = np.random.randint(1, 32)

random_mri_prob_bringer(image_number=rand_img_no)

Comparing Predicted Classes with the Actual Classes from the Test Data

plt.figure(figsize=(20, 20), facecolor="gray")

for images, labels in test_data.take(1):

for i in range(25):

ax = plt.subplot(5, 5, i + 1)

plt.imshow(images[i])

predictions = model.predict(tf.expand_dims(images[i], 0), verbose=0)

score = tf.nn.softmax(predictions[0])

if(class_names[labels[i]]==class_names[np.argmax(score)]):

plt.title("Actual: "+class_names[labels[i]], color="aqua", fontweight="bold",

fontsize=10)

plt.ylabel("Predicted: "+class_names[np.argmax(score)], color="springgreen",

fontweight="bold", fontsize=10)

ok_text = plt.text(2, 10, "OK \u2714", color="springgreen", fontsize=14)

ok_text.set_bbox(dict(facecolor='lime', alpha=0.5))

else:

plt.title("Actual: "+class_names[labels[i]], color="aqua", fontweight="bold",

fontsize=10)

plt.ylabel("Predicted: "+class_names[np.argmax(score)], color="maroon",

fontweight="bold", fontsize=10)

nok_text = plt.text(2, 10, "NOK \u2718", color="red", fontsize=14)

nok_text.set_bbox(dict(facecolor='maroon', alpha=0.5))

plt.gca().axes.yaxis.set_ticklabels([])

plt.gca().axes.xaxis.set_ticklabels([])# Project-phase-1
```
## Output:
![image](https://github.com/user-attachments/assets/b62c3fb8-a2aa-450f-99bf-cfed171759f4)


#### Output1 - Name of the output

![image](https://github.com/user-attachments/assets/e1ebf68f-cf2c-4d50-892e-7f06d2921e9e)

#### Output2 - Name of the output
![image](https://github.com/user-attachments/assets/de4a8979-e289-408a-9631-e0a9d630029a)

Detection Accuracy: 93%

![image](https://github.com/user-attachments/assets/e80e1313-b7f5-451c-874e-1ec9934ffeba)

## Results and Impact
The Alzheimer’s disease detection project using deep learning aims to accurately identify early signs of Alzheimer's from brain scan images. By training a model on MRI or PET scan data, the system can distinguish between healthy brains and those affected by Alzheimer's, potentially improving early diagnosis. 

This result provides clinicians with an advanced tool to support decision-making, ultimately helping with timely intervention and potentially slowing disease progression. The impact of this project extends to improving patient care and advancing Alzheimer's research by offering data-driven insights for diagnosis and treatment.
## Articles published / References
1. Ferreira L K, Rondina Luiz et al 2017 Support vector machine-based classification of neuroimages in Alzheimer's disease: Direct comparison of FDG-PET, rCBF-SPECT and MRI data acquired from the same
   individuals Braz J. of Psychiatry 40 181-91
2. Zhu Yingying, Zhu Xiaofeng et al 2016 Early Diagnosis of Alzheimer's Disease by Joint Featureselection and classification on temporally Structured Support Vector Machine Med Image Comp Assist
   Interv 264-72




