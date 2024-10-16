#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Supported by BiCubes project (HFRI grant 3943)

"""
This script performs image classification using transfer learning with a pretrained ResNet50 model.
It fine-tunes the model on a custom dataset and evaluates its performance.

Dependencies:
- numpy
- tensorflow
- keras
- sklearn
- argparse
- os
"""

import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, optimizers
from sklearn.metrics import classification_report, confusion_matrix
import argparse

def load_data(data_path, img_size=(224, 224), batch_size=32):
  """
  Load and preprocess image data using Keras ImageDataGenerator.

  Parameters:
  - data_path: str, path to the directory containing images.
  - img_size: tuple, size to which images will be resized.
  - batch_size: int, number of samples per batch.

  Returns:
  - train_generator: DirectoryIterator, training data generator.
  - validation_generator: DirectoryIterator, validation data generator.
  """
  datagen = ImageDataGenerator(validation_split=0.2, rescale=1./255)

  train_generator = datagen.flow_from_directory(
      data_path,
      target_size=img_size,
      batch_size=batch_size,
      class_mode='categorical',
      subset='training'
  )

  validation_generator = datagen.flow_from_directory(
      data_path,
      target_size=img_size,
      batch_size=batch_size,
      class_mode='categorical',
      subset='validation'
  )

  return train_generator, validation_generator

def build_model(input_shape, num_classes):
  """
  Build a CNN model using a pretrained ResNet50 as the base.

  Parameters:
  - input_shape: tuple, shape of the input data.
  - num_classes: int, number of classes for classification.

  Returns:
  - model: Keras Model, the CNN model.
  """
  base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
  base_model.trainable = False  # Freeze the base model

  model = models.Sequential([
      base_model,
      layers.GlobalAveragePooling2D(),
      layers.Dense(1024, activation='relu'),
      layers.Dropout(0.5),
      layers.Dense(num_classes, activation='softmax')
  ])

  return model

def main(args):
  # Load and preprocess data
  train_generator, validation_generator = load_data(args.data_path, img_size=(args.img_size, args.img_size), batch_size=args.batch_size)

  # Build and compile the model
  model = build_model(input_shape=(args.img_size, args.img_size, 3), num_classes=args.num_classes)
  model.compile(optimizer=optimizers.Adam(learning_rate=args.learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])

  # Train the model
  model.fit(train_generator, epochs=args.epochs, validation_data=validation_generator)

  # Evaluate the model
  validation_generator.reset()
  predictions = model.predict(validation_generator)
  y_pred = np.argmax(predictions, axis=1)
  y_true = validation_generator.classes

  print(classification_report(y_true, y_pred, target_names=validation_generator.class_indices.keys()))
  print(confusion_matrix(y_true, y_pred))

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description="Train a CNN using transfer learning with ResNet50.")
  parser.add_argument("-d", "--data-path", type=str, required=True, help="Path to the directory containing image data.")
  parser.add_argument("-e", "--epochs", type=int, default=10, help="Number of epochs for training.")
  parser.add_argument("-b", "--batch-size", type=int, default=32, help="Batch size for training.")
  parser.add_argument("-s", "--img-size", type=int, default=224, help="Size to which images will be resized (img_size x img_size).")
  parser.add_argument("-c", "--num-classes", type=int, required=True, help="Number of classes for classification.")
  parser.add_argument("-lr", "--learning-rate", type=float, default=0.0001, help="Learning rate for the optimizer.")
  args = parser.parse_args()

  main(args)
