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
        
class TimingCallback(callbacks.Callback):
    def __init__(self):
        super(TimingCallback, self).__init__()
        self.startTime = None
        self.epochStart = None
        self.totalTime = None
        self.precision_metric = Precision()
        self.recall_metric = Recall()
        self.accuracy_metric = BinaryAccuracy()
        self.epoch_count = 0

    def format_time(seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = seconds % 60
        return f'{hours}h {minutes}m {seconds:.2f}s'

    def on_train_begin(self, logs=None):
        self.totalTime_start = time.time()

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_count += 1
        self.epoch_start = time.time()

    def on_batch_begin(self, batch, logs=None):
        self.batch_start = time.time()

    def on_batch_end(self, batch, logs=None):
        batch_time = time.time() - self.batch_start
        
        if 'y_true' in logs and 'y_pred' in logs:
            y_true = logs['y_true']
            y_pred = logs['y_pred']

            self.precision_metric.update_state(y_true, y_pred)
            self.recall_metric.update_state(y_true, y_pred)
            self.accuracy_metric.update_state(y_true, y_pred)

            loss = logs.get('loss', 'N/A')
            accuracy = logs.get('accuracy', 'N/A')
            precision = logs.get('precision', 'N/A')
            recall = logs.get('recall', 'N/A')
            
            print('-'*70)
            print(f'Epoch {self.epoch_count}\tBatch {batch}')
            print(f'Precision: {precision:.4f}\tRecall: {recall:.4f}\nAccuracy: {accuracy:.4f}\tLoss: {loss}')
            print(f'Batch: {self.format_time(batch_time)}')
            print('-'*70)

    def on_epoch_end(self, epoch, logs=None):
        epoch_time = time.time() - self.epoch_start
        print(f'Epoch {epoch}: {self.format_time(epoch_time)}')

    def on_train_end(self, logs=None):
        totalTime = time.time() - self.totalTime_start
        print(f'Total training: {self.format_time(totalTime)}')