from tensorflow import keras
from keras import callbacks
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy # type: ignore
import time
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