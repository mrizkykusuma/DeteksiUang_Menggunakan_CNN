# -*- coding: utf-8 -*-
"""DeteksiUang_menggunakan_CNN.py

Klasifikasi Gambar Uang Dengan Convolusional Neural Network (CNN)
"""

import os
import random
import numpy as np
import matplotlib.pyplot as plt
from shutil import copyfile
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.preprocessing import image

# Function to dynamically find directories
def find_directories(base_path, dir_names):
    dirs = {}
    for root, subdirs, files in os.walk(base_path):
        for name in dir_names:
            if name in subdirs:
                dirs[name] = os.path.join(root, name)
    return dirs

# Define the base path
base_path = 'D:\kuliah\semester 6\Pengolahan Citra Digital\Deteksi Uang menggunakan CNN\kelompok2'  # Set your base directory path here
dir_names = ['bahan', 'latih', 'validasi']

# Find directories
directories = find_directories(base_path, dir_names)
bahan_dir = directories.get('bahan', '')
train_dir = directories.get('latih', '')
validation_dir = directories.get('validasi', '')

# Check if directories are found
if not bahan_dir or not train_dir or not validation_dir:
    raise ValueError("One or more required directories are missing.")

# Define data directories
limaribu_dir = os.path.join(bahan_dir, 'limaribu/')
sepuluhribu_dir = os.path.join(bahan_dir, 'sepuluhribu/')
train_limaribu = os.path.join(train_dir, 'limaribu/')
train_sepuluhribu = os.path.join(train_dir, 'sepuluhribu/')
validation_limaribu = os.path.join(validation_dir, 'limaribu/')
validation_sepuluhribu = os.path.join(validation_dir, 'sepuluhribu/')

# Print number of images in each class
print("Jumlah data train tiap kelas")
print('Jumlah gambar uang 5.000 :', len(os.listdir(limaribu_dir)))
print('Jumlah gambar uang 10.000 :', len(os.listdir(sepuluhribu_dir)))

def train_val_split(source, train, val, train_ratio):
    total_size = len(os.listdir(source))
    train_size = int(train_ratio * total_size)
    val_size = total_size - train_size

    randomized = random.sample(os.listdir(source), total_size)
    train_files = randomized[0:train_size]
    val_files = randomized[train_size:total_size]

    for i in train_files:
        i_file = os.path.join(source, i)
        destination = os.path.join(train, i)
        copyfile(i_file, destination)

    for i in val_files:
        i_file = os.path.join(source, i)
        destination = os.path.join(val, i)
        copyfile(i_file, destination)

# Split data into training and validation sets
train_ratio = 0.8
train_val_split(limaribu_dir, train_limaribu, validation_limaribu, train_ratio)
train_val_split(sepuluhribu_dir, train_sepuluhribu, validation_sepuluhribu, train_ratio)

# Print number of images after split
print("Jumlah All limaribu :", len(os.listdir(limaribu_dir)))
print("Jumlah Train limaribu :", len(os.listdir(train_limaribu)))
print("Jumlah Val limaribu :", len(os.listdir(validation_limaribu)))
print("")
print("Jumlah All sepuluhribu :", len(os.listdir(sepuluhribu_dir)))
print("Jumlah Train sepuluhribu :", len(os.listdir(train_sepuluhribu)))
print("Jumlah Val sepuluhribu :", len(os.listdir(validation_sepuluhribu)))

# Image data generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    horizontal_flip=True,
    shear_range=0.3,
    fill_mode='nearest',
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.1
)

val_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    horizontal_flip=True,
    shear_range=0.3,
    fill_mode='nearest',
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.1
)

# Generators
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=10,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=10,
    class_mode='categorical'
)

class myCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs.get('accuracy') > 0.99:
            print('\nAccuracy has reached 99%')
            self.model.stop_training = True

callbacks = myCallback()

# Define CNN model
model = models.Sequential([
    layers.Conv2D(16, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(200, activation='relu'),
    layers.Dropout(0.3, seed=112),
    layers.Dense(500, activation='relu'),
    layers.Dropout(0.5, seed=112),
    layers.Dense(2, activation='sigmoid')
])

model.summary()

model.compile(
    loss='categorical_crossentropy',
    optimizer='Adam',
    metrics=['accuracy']
)

history = model.fit(
    train_generator,
    steps_per_epoch=2,
    epochs=25,
    validation_data=val_generator,
    validation_steps=1,
    verbose=1,
    callbacks=[callbacks]
)

# Plot training and validation accuracy/loss
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training Accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend(loc='best')
plt.show()

plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend(loc='best')
plt.show()

# Predicting new images
uploaded_files = ['path_to_your_test_image.jpg']  # Replace with your test image paths
for fn in uploaded_files:
    img = image.load_img(fn, target_size=(150, 150))
    imgplot = plt.imshow(img)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    images = np.vstack([x])
    classes = model.predict(images, batch_size=10)

    print(fn)
    class_list = os.listdir(train_dir)
    for j in range(len(classes[0])):
        if classes[0][j] == 1.:
            print('Gambar ini masuk ke kelas', class_list[j-1])
            break
