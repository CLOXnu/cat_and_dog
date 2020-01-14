
# %% import
import tensorflow as tf
import numpy as np
from tensorflow.keras import models, layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import matplotlib.pyplot as plt


# %%
data_path = 'datas/'
train_path = os.path.join(data_path, 'train')
val_path = os.path.join(data_path, 'validation')

train_cats_path = os.path.join(train_path, 'cats')
train_dogs_path = os.path.join(train_path, 'dogs')
val_cats_path = os.path.join(val_path, 'cats')
val_dogs_path = os.path.join(val_path, 'dogs')


# %%
num_train_cats = len(os.listdir(train_cats_path))
num_train_dogs = len(os.listdir(train_dogs_path))

num_val_cats = len(os.listdir(val_cats_path))
num_val_dogs = len(os.listdir(val_dogs_path))

num_train = num_train_cats + num_train_dogs
num_val = num_val_cats + num_val_dogs

print('train cats:', num_train_cats)
print('train dogs:', num_train_dogs)
print('validation cats:', num_val_cats)
print('validation dogs:', num_val_dogs)
print('train:', num_train)
print('validation:', num_val)


# %%
batch_size = 128
epochs = 15
IMG_HEIGHT = 150
IMG_WIDTH = 150


# %%
train_image_generator = ImageDataGenerator(
    rescale=1./255,
    rotation_range=45,
    width_shift_range=.15,
    height_shift_range=.15,
    horizontal_flip=True,
    zoom_range=0.5)
val_image_generator = ImageDataGenerator(rescale=1./255)

train_data_gen = train_image_generator.flow_from_directory(
    batch_size=batch_size,             
    directory=train_path, 
    shuffle=True,   
    target_size=(IMG_HEIGHT, IMG_WIDTH), 
    class_mode='binary')

val_data_gen = val_image_generator.flow_from_directory(
    batch_size=batch_size,             
    directory=val_path, 
    target_size=(IMG_HEIGHT, IMG_WIDTH), 
    class_mode='binary')


# %%
checkpoint_path = 'training1/cp-{epoch:04d}.ckpt'
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    verbose=1,
    save_weights_only=True,
    save_freq=5
)


# %%
def create_model():
    model = models.Sequential([
        layers.Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy'])
    return model


# %%
model = create_model()
model.summary()


# %%
model_log = model.fit_generator(
    train_data_gen,
    steps_per_epoch=num_train,
    epochs=epochs,
    callbacks=[cp_callback],
    validation_data=val_data_gen,
    validation_steps=num_val
)


# %%
acc = model_log.history['accuracy']
val_acc = model_log.history['val_accuracy']

loss = model_log.history['loss']
val_loss = model_log.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Loss')
plt.savefig('A&L.jpg')
plt.show()


# %%
model.save('model.h5')

