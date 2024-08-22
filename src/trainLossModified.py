import numpy as np
import sys
import argparse
from math import ceil
import os
import time
from os import listdir, makedirs
from os.path import join, exists, splitext
from PIL import Image
from sklearn import preprocessing
import random
from mCNN import createModel
import tensorflow as tf
from tensorflow import keras
from keras.utils import to_categorical # type: ignore
from keras.callbacks import ModelCheckpoint # type: ignore

def custom_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    # Cálculo de la pérdida estándar
    loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred, from_logits=False)
    
    # Penalización para falsos positivos cuando y_true es 0 y y_pred es 1
    false_positives = tf.reduce_sum((1 - y_true) * y_pred, axis=-1)
    
    return loss + penalty_factor * false_positives

optimizer = keras.optimizers.Adam(learning_rate=1e-3)
loss_fn = custom_loss
train_acc_metric = keras.metrics.CategoricalAccuracy()
val_acc_metric = keras.metrics.CategoricalAccuracy()

penalty_factor = 2.5

def main(args):
    global penalty_factor, optimizer
    positiveImagePath = args.positiveImages
    negativeImagePath = args.negativeImages
    
    positiveDataImagePath = args.trainingDataPositive
    negativeDataImagePath = args.trainingDataNegative
    
    numEpochs = args.epochs
    save_epoch = args.save_epoch
    init_epoch = args.init_epoch
    save_iter = args.save_iter
    
    batch_size = args.batch_size
    
    checkpointPath = args.checkpointPath
    loadCheckPoint = args.loadCheckPoint
    
    height = args.height
    width = args.width
    
    penalty_factor = args.penalty_factor
    learning_rate = args.learning_rate
    
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        
    trainIndex, numClasses = createIndex(positiveImagePath, negativeImagePath)

    epochFilePath = f"{checkpointPath}/epoch.txt"
    checkpoint_path = f"{checkpointPath}/cp.keras"
    
    if not exists(checkpointPath):
        makedirs(checkpointPath)
    
    model = createModel(height=height, width=width, depth=1, num_classes=numClasses)

    if loadCheckPoint:
        model.load_weights(checkpoint_path)

    model.compile(
        loss=custom_loss,
        optimizer='adam',
        metrics=['accuracy']
)
    
    epoch = epochFileValidation(epochFilePath, loadCheckPoint, init_epoch)
    
    model = trainModel(trainIndex, positiveDataImagePath, negativeDataImagePath, epoch, numEpochs, 
        epochFilePath, save_epoch, save_iter, batch_size, numClasses, height, width, checkpoint_path, model)

def readAndScaleImage(f, customStr, trainImagePath, X_LL, X_LH, X_HL, X_HH, X_index, Y, sampleIndex, sampleVal, height, width):
    f = str(f)
    fileName = (os.path.splitext(f)[0])
    fLL = f"{splitext(f.replace(fileName, fileName + customStr + '_LL'))}.tiff"
    fLH = (f.replace(fileName, fileName + customStr + '_LH')).replace('.jpg','.tiff')
    fHL = (f.replace(fileName, fileName + customStr + '_HL')).replace('.jpg','.tiff')
    fHH = (f.replace(fileName, fileName + customStr + '_HH')).replace('.jpg','.tiff')
    
    try:
        imgLL = PreprocessImage(join(trainImagePath, fLL), width, height)
        imgLL = PreprocessImage(join(trainImagePath, fLH), width, height)
        imgLL = PreprocessImage(join(trainImagePath, fHL), width, height)
        imgLL = PreprocessImage(join(trainImagePath, fHH), width, height)
        
        # DATA AUGMENTATION FOR TRAINING
        imgLL = apply_augmentation(imgLL)
        imgLH = apply_augmentation(imgLH)
        imgHL = apply_augmentation(imgHL)
        imgHH = apply_augmentation(imgHH)
        
    except Exception as e:
        print(f"Error: Couldn\'t read the file {fileName}. Make sure only images are present in the folder")
        print('Exception:', e)
        return None

    imgLL = np.array(imgLL)
    imgLH = np.array(imgLH)
    imgHL = np.array(imgHL)
    imgHH = np.array(imgHH)
    imgLL = scaleData(imgLL, 0, 1)
    imgLH = scaleData(imgLH, -1, 1)
    imgHL = scaleData(imgHL, -1, 1)
    imgHH = scaleData(imgHH, -1, 1)

    imgVector = imgLL.reshape(1, width*height)
    X_LL[sampleIndex, :] = imgVector
    imgVector = imgLH.reshape(1, width*height)
    X_LH[sampleIndex, :] = imgVector
    imgVector = imgHL.reshape(1, width*height)
    X_HL[sampleIndex, :] = imgVector
    imgVector = imgHH.reshape(1, width*height)
    X_HH[sampleIndex, :] = imgVector
    
    Y[sampleIndex, 0] = sampleVal
    X_index[sampleIndex, 0] = sampleIndex

    return True

def PreprocessImage(imgPath, width, height):
    img = Image.open(imgPath)
    w, h = img.size
    
    if w < width or h < height:
        proportion = min(width / w, height / h)
        new_width = int(w * proportion)
        new_height = int(h * proportion)
        
        img = img.resize((new_width, new_height), Image.LANCZOS)
        
        left = (new_width - width) / 2
        top = (new_height - height) / 2
        right = (new_width + width) / 2
        bottom = (new_height + height) / 2
        
        img = img.crop((left, top, right, bottom))
    
    elif  w > width or h > height:
        proportion = max(width / w, height / h)
        new_width = int(w * proportion)
        new_height = int(h * proportion)
        
        img = img.resize((new_width, new_height), Image.LANCZOS)
        
        left = (new_width - width) / 2
        top = (new_height - height) / 2
        right = (new_width + width) / 2
        bottom = (new_height + height) / 2
        
        img = img.crop((left, top, right, bottom))
    
    return img.convert('L')

def apply_augmentation(image):
    image_np = np.array(image)
    
    if len(image_np.shape) != 2:
        raise ValueError("The input image must be a 2-D grayscale image.")
    
    image_tf = tf.convert_to_tensor(image_np, dtype=tf.float32)
    image_tf = tf.expand_dims(image_tf, axis=-1)
    
    image_tf = tf.image.random_brightness(image_tf, max_delta=0.15)
    image_tf = tf.image.random_contrast(image_tf, lower=0.9, upper=1.1)
    
    image_tf = tf.squeeze(image_tf, axis=-1)
    image_tf = tf.clip_by_value(image_tf, 0, 255)
    image_tf = tf.cast(image_tf, dtype=tf.uint8)
    
    image_augmented = Image.fromarray(image_tf.numpy())
    
    return image_augmented

def scaleData(inp, minimum, maximum):
    minMaxScaler = preprocessing.MinMaxScaler(copy=True, feature_range=(minimum,maximum))
    inp = inp.reshape(-1, 1)
    inp = minMaxScaler.fit_transform(inp)
    
    return inp

def createIndex(posPath, negPath):
    posList = list(listdir(posPath))
    negList = list(listdir(negPath))

    datasetList = [(i, 1) for i in posList]
    datasetList.extend((i, 0) for i in negList)

    random.shuffle(datasetList)
    
    classes = len(np.unique(np.array([fila[1] for fila in datasetList])))
    
    return datasetList, classes

def epochFileValidation(path, loadFlag, init_epoch):
    if not exists(path) or not loadFlag:
        with open(path, 'w') as epochFile:
            if init_epoch == 0:
                epoch = 1
                epochFile.write("1")
            else:
                epoch = init_epoch
                epochFile.write(str(epoch))
    elif init_epoch == 0:
        with open(path, 'r') as epochFile:
            epoch = int(epochFile.read())
    else:
        epoch = init_epoch

    return epoch

def saveEpochFile(epochFilePath, epoch):
    with open(epochFilePath, 'w') as epochFile:
        epochFile.write(str(epoch))
        print(f"\nEpoch Save: {epoch}")
        
@tf.function
def train_step(model, X_LL_train, X_LH_train, X_HL_train, X_HH_train, Y_train):
    with tf.GradientTape() as tape:
        logits = model([X_LL_train, X_LH_train, X_HL_train, X_HH_train], training=True)  # Logits for this minibatch
        loss_value = tf.reduce_mean(loss_fn(Y_train, logits))

    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    train_acc_metric.update_state(Y_train, logits)
    
    return loss_value
        
def trainModel(listInput, posPath, negPath, epoch, epochs, epochFilePath, save_epoch, save_iter, batch_size, numClasses, height, width, checkpoint_path, model):
    epoch -= 1
    n = len(listInput)
    start_time_full = time.time()
    for i in range(epochs - epoch):
        saveEpochFile(epochFilePath, i + 1)
        print(f"epoch: {i + epoch + 1}/{epochs}\n")
        start_time = time.time()
        for j in range(ceil(n/batch_size)):
            start, end = defineEpochRange(j, batch_size, n)            
            X_LL_train, X_LH_train, X_HL_train, X_HH_train, Y_train = getBatch(listInput, posPath, negPath, start, end, batch_size, height, width)
            Y_train = to_categorical(Y_train, numClasses)
            
            loss = train_step(model, X_LL_train, X_LH_train, X_HL_train, X_HH_train, Y_train)
            
            print("------------------------------------")
            print(f"Training {end - start} images ({j + 1}/{ceil(n/batch_size)})", end='\t')
            print(f'start: {start}\tend: {end}\tTotal Images:{len(listInput)}\tLoss: {loss*100:.2f}%')
            train_acc = train_acc_metric.result()
            print(f'\nTraining acc over batch: {float(train_acc.numpy())*100:.2f}%')

            # Reset training metrics at the end of each epoch
            train_acc_metric.reset_state()

            # Run a validation loop at the end of each epoch.
            val_logits = model([X_LL_train, X_LH_train, X_HL_train, X_HH_train], training=False)
            # Update val metrics
            val_acc_metric.update_state(Y_train, val_logits)
            val_acc = val_acc_metric.result()
            print(f'Validation acc: {float(val_acc.numpy())*100:.2f}%')
            val_acc_metric.reset_state()
            
            if save_iter != 0 and (j + 1) % save_iter == 0:
                saveModel(model, checkpoint_path)
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        minutes = elapsed_time // 60
        remaining_seconds = elapsed_time % 60
        print(f"Batch time: {int(minutes)} minutes {remaining_seconds:.2f} seconds")

        if (i + 1) % save_epoch == 0:
            saveModel(model, checkpoint_path)
        print("------------------------------------")
    saveModel(model, checkpoint_path)

    end_time_full = time.time()
    elapsed_time = end_time_full - start_time_full
    hours = elapsed_time // 3600
    elapsed_time %= 3600
    minutes, seconds = divmod(elapsed_time, 60)
    print(f"\nTotal training time: {int(hours)} hours, {int(minutes)} minutes, {seconds:.2f} seconds.")

    return model
        
def saveModel(model, checkpoint_path):
    print("Saving model... ", end='')
    model.save(checkpoint_path, include_optimizer=False)
    print("Model Saved.")
            

def createElements(batch_size, height, width, multiply):
    totalBatchSize = batch_size*multiply
    X_LL = np.zeros((totalBatchSize, width*height))
    X_LH = np.zeros((totalBatchSize, width*height))
    X_HL = np.zeros((totalBatchSize, width*height))
    X_HH = np.zeros((totalBatchSize, width*height))
    X_index = np.zeros((totalBatchSize, 1))
    Y = np.zeros((totalBatchSize, 1))
    
    #return X_LL.astype(np.float32), X_LH.astype(np.float32), X_HL.astype(np.float32), X_HH.astype(np.float32), Y
    return X_LL, X_LH, X_HL, X_HH, X_index, Y, totalBatchSize

def defineEpochRange(epoch, batch_size, n):
    start = 0 if epoch*batch_size >= n else epoch*batch_size
    end = min(start + batch_size, n)
    
    return start, end

def getBatch(listInput, posPath, negPath, start, end, batch_size, height, width):
    X_LL, X_LH, X_HL, X_HH, X_index, Y, totalBatchSize= createElements(batch_size, height, width, 3)

    sampleIndex = 0
    for f in listInput[start:end]:
        file = str(f[0])
        y = int(f[1])
        path = posPath if y == 1 else negPath
        
        ret = readAndScaleImage(file, '', path, X_LL, X_LH, X_HL, X_HH, X_index, Y, sampleIndex, y, height, width)
        if ret == True:
            sampleIndex += 1

        #read 180deg rotated data
        ret = readAndScaleImage(file, '_180', path, X_LL, X_LH, X_HL, X_HH, X_index, Y, sampleIndex, y, height, width)
        if ret == True:
            sampleIndex += 1

        #read 180deg FLIP data
        ret = readAndScaleImage(file, '_180_FLIP', path, X_LL, X_LH, X_HL, X_HH, X_index, Y, sampleIndex, y, height, width)
        if ret == True:
            sampleIndex += 1

    X_LL = X_LL.reshape((totalBatchSize, height, width, 1))
    X_LH = X_LH.reshape((totalBatchSize, height, width, 1))
    X_HL = X_HL.reshape((totalBatchSize, height, width, 1))
    X_HH = X_HH.reshape((totalBatchSize, height, width, 1))
    
    return X_LL, X_LH, X_HL, X_HH, Y

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1', 'True'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0', 'False'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--positiveImages', type=str, help='Directory with original positive (Moiré pattern) images.')
    parser.add_argument('--negativeImages', type=str, help='Directory with original negative (Normal) images.')
    
    parser.add_argument('--trainingDataPositive', type=str, help='Directory with transformed positive (Moiré pattern) images.')
    parser.add_argument('--trainingDataNegative', type=str, help='Directory with transformed negative (Normal) images.')
    
    parser.add_argument('--checkpointPath', type=str, help='Directory for model Checkpoint', default='./checkpoint/')
    
    parser.add_argument('--epochs', type=int, help='Number of epochs for training', default=10)
    parser.add_argument('--save_epoch', type=int, help='Number of epochs to save the model', default=10)
    parser.add_argument('--init_epoch', type=int, help='Initial epoch for model training (if 0, starts with the saved epoch file)', default=0)
    parser.add_argument('--save_iter', type=int, help='Number of iterations to save the model', default=0)

    parser.add_argument('--batch_size', type=int, help='Batch size for epoch in training', default=32)
    parser.add_argument('--height', type=int, help='Image height resize', default=800)
    parser.add_argument('--width', type=int, help='Image width resize', default=1400)
    
    parser.add_argument('--penalty_factor', type=float, help='Penalty to False Positives predictions', default=2.5)
    parser.add_argument('--learning_rate', type=float, help='Model learning rate for iteration', default=1e-3)
    
    parser.add_argument('--loadCheckPoint', type=str2bool, help='Enable Checkpoint Load', default='True')
    
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))