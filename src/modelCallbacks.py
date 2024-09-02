import tensorflow as tf
from tensorflow import keras
from keras import callbacks
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy # type: ignore
import time
import os
import numpy as np

class BatchCheckpointCallback(callbacks.Callback):
    def __init__(self, batchesNumber, path):
        super(BatchCheckpointCallback, self).__init__()
        self.count = 0
        self.batchesNumber = batchesNumber
        self.modelSaveBatchPath = path

    def on_batch_end(self, batch, logs=None):
        self.count += 1
        if self.count % self.batchesNumber == 0:
            print('\nGuardando modelo... ', end='')
            self.model.save(self.modelSaveBatchPath)
            print(f'Modelo guardado en {self.modelSaveBatchPath}')
            
class EpochCheckpointCallback(callbacks.Callback):
    def __init__(self, path):
        super(EpochCheckpointCallback, self).__init__()
        self.modelSaveEpochPath = path

    def on_epoch_end(self, epoch, logs=None):
        print('Guardando modelo... ', end='')
        self.model.save(self.modelSaveEpochPath)
        print(f'Modelo guardado en {self.modelSaveEpochPath}')
        
class CustomImageDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, directory, batch_size, image_size, preprocess_function, class_mode='binary', classes=None):
        self.directory = directory
        self.batch_size = batch_size
        self.image_size = image_size
        self.preprocess_function = preprocess_function
        self.class_mode = class_mode
        self.classes = classes
        self.image_paths = self._get_image_paths()
        self.indexes = np.arange(len(self.image_paths))

    def _get_image_paths(self):
        image_extensions = ('.jpg', '.jpeg', '.png')
        image_paths = []
        for root, _, files in os.walk(self.directory):
            for file in files:
                if file.lower().endswith(image_extensions):
                    image_paths.append(os.path.join(root, file))
        return image_paths

    def __len__(self):
        return int(np.ceil(len(self.image_paths) / self.batch_size))

    def __getitem__(self, index):
        batch_image_paths = self.image_paths[index * self.batch_size:(index + 1) * self.batch_size]
        batch_images_dict = {
            'LL_Input': [],
            'LH_Input': [],
            'HL_Input': [],
            'HH_Input': [],
            'Scharr_Input': []  # Añadir Scharr_Input
        }
        
        for image_path in batch_image_paths:
            components = self._load_and_preprocess_image(image_path)
            for key in batch_images_dict:
                batch_images_dict[key].append(components[key])
        
        batch_images_dict = {key: np.stack(value) for key, value in batch_images_dict.items()}
        
        if self.class_mode == 'binary':
            batch_labels = np.array([self.classes[os.path.basename(os.path.dirname(image_path))] for image_path in batch_image_paths])
            return batch_images_dict, batch_labels
        else:
            raise ValueError("Unsupported class_mode")

    def _load_and_preprocess_image(self, image_path):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_image(image, channels=3)
        image = tf.image.resize(image, self.image_size)
        components = self.preprocess_function(image)
        
        # Verificar que la función de preprocesamiento devuelve un diccionario con las claves correctas
        required_keys = {'LL_Input', 'LH_Input', 'HL_Input', 'HH_Input', 'Scharr_Input'}
        if not all(key in components for key in required_keys):
            raise ValueError("Preprocessing function must return a dictionary with keys 'LL_Input', 'LH_Input', 'HL_Input', 'HH_Input', and 'Scharr_Input'.")
        
        return components