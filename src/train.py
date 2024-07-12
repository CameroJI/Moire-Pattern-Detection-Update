from matplotlib import pyplot as plt
import numpy as np
import sys
import argparse
from math import ceil
import os
import time
from os import listdir, makedirs
from os.path import join, exists
from PIL import Image
from sklearn import preprocessing
import random
from mCNN import createModel
import tensorflow
from tensorflow import keras
from keras.utils import to_categorical # type: ignore
from keras.callbacks import ModelCheckpoint # type: ignore

optimizer = keras.optimizers.SGD(learning_rate=1e-3)
loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
train_acc_metric = keras.metrics.SparseCategoricalAccuracy()
val_acc_metric = keras.metrics.SparseCategoricalAccuracy()

def main(args):
    positiveImagePath = (args.positiveImages)
    negativeImagePath = (args.negativeImages)
    numEpochs = (args.epochs)
    save_epoch = (args.save_epoch)
    positiveDataImagePath = args.trainingDataPositive
    negativeDataImagePath = args.trainingDataNegative
    batch_size = (args.batch_size)
    checkpointPath = (args.checkpointPath)
    loadCheckPoint = (args.loadCheckPoint)
    height = (args.height)
    width = (args.width)
    
    trainIndex, valIndex, numClasses = createIndex(positiveImagePath, negativeImagePath)

    epochFilePath = f"{checkpointPath}/epoch.txt"
    checkpoint_path = f"{checkpointPath}/cp.h5"
    if not exists(checkpointPath):
        makedirs(checkpointPath)
        
    model = createModel(height=height, width=width, depth=1, num_classes=numClasses)
    model.compile(loss='categorical_crossentropy', # using the cross-entropy loss function
        optimizer='adam', # using the Adam optimizer
        metrics=['accuracy']) # reporting the accuracy 

    if loadCheckPoint:
        model.load_weights(checkpoint_path)
    
    epoch = epochFileValidation(epochFilePath, loadCheckPoint)
    
    model = trainModel(trainIndex, positiveDataImagePath, negativeDataImagePath, epoch, numEpochs, 
        epochFilePath, save_epoch, batch_size, numClasses, height, width, checkpoint_path, model)

    evaluate(model, valIndex, positiveDataImagePath, negativeDataImagePath, height, width)


def readAndScaleImage(f, customStr, trainImagePath, X_LL, X_LH, X_HL, X_HH, X_index, Y, sampleIndex, sampleVal, height, width):
    f = str(f)
    fileName = (os.path.splitext(f)[0])
    fLL = (f.replace(fileName, fileName + customStr + '_LL')).replace('.jpg','.tiff')
    fLH = (f.replace(fileName, fileName + customStr + '_LH')).replace('.jpg','.tiff')
    fHL = (f.replace(fileName, fileName + customStr + '_HL')).replace('.jpg','.tiff')
    fHH = (f.replace(fileName, fileName + customStr + '_HH')).replace('.jpg','.tiff')
    
    try:
        imgLL = Image.open(join(trainImagePath, fLL))
        imgLH = Image.open(join(trainImagePath, fLH))
        imgHL = Image.open(join(trainImagePath, fHL))
        imgHH = Image.open(join(trainImagePath, fHH))
        
        imgLL = imgLL.resize((width, height))
        imgLH = imgLH.resize((width, height))
        imgHL = imgHL.resize((width, height))
        imgHH = imgHH.resize((width, height))
        
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

    #dataset split for training and evaluation
    trainList = []
    valList = []

    random.shuffle(datasetList)
    for i in range(len(datasetList)):
        if i < len(datasetList) - int(len(datasetList)*0.1):
            trainList.append(datasetList[i])
        else:
            valList.append(datasetList[i])
    
    classes = len(np.unique(np.array([fila[1] for fila in datasetList])))
    
    return trainList, valList, classes

def epochFileValidation(path, loadFlag):
    if not exists(path) or not loadFlag:
        with open(path, 'w') as epochFile:
            epochFile.write("1")
            epoch = 1
    else:
        with open(path, 'r') as epochFile:
            epoch = int(epochFile.read())
            
    return epoch

def saveEpochFile(epochFilePath, epoch):
    with open(epochFilePath, 'w') as epochFile:
        epochFile.write(str(epoch + 1))
        print(f"\nEpoch Save: {epoch + 1}")
        
@tensorflow.function
def train_step(model, X_LL_train, X_LH_train, X_HL_train, X_HH_train, Y_train):
    logits = model([X_LL_train, X_LH_train, X_HL_train, X_HH_train], training=True)  # Logits for this minibatch
    with tensorflow.GradientTape() as tape:
        logits = model([X_LL_train, X_LH_train, X_HL_train, X_HH_train], training=True)  # Logits for this minibatch
        loss_value = loss_fn(Y_train, logits)

    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    train_acc_metric.update_state(Y_train, logits)
    
    return loss_value
        
def trainModel(listInput, posPath, negPath, epoch, epochs, epochFilePath, save_epoch, batch_size, numClasses, height, width, checkpoint_path, model):
    epoch -= 1
    n = len(listInput)

    for i in range(epochs - epoch):
        print(f"epoch: {i + 1}/{epochs}\n")
        start_time = time.time()
        for j in range(ceil(n/batch_size)):
            start, end = defineEpochRange(j, batch_size, n)            
            X_LL_train, X_LH_train, X_HL_train, X_HH_train, Y_train = getBatch(listInput, posPath, negPath, start, end, batch_size, height, width)
            #Y_train = to_categorical(Y_train, numClasses)

            loss = train_step(model, X_LL_train, X_LH_train, X_HL_train, X_HH_train, Y_train)
            print("------------------------------------")
            print(f"Training {end - start} images.", end='\t')
            print(f'start: {start}\tend: {end}\tTotal Images:{len(listInput)}\t', end='Loss: ')
            print(float(loss))
            train_acc = train_acc_metric.result()
            print("\nTraining acc over batch: %.4f" % (float(train_acc),))

            # Reset training metrics at the end of each epoch
            train_acc_metric.reset_states()

            # Run a validation loop at the end of each epoch.
            val_logits = model([X_LL_train, X_LH_train, X_HL_train, X_HH_train], training=False)
                # Update val metrics
            val_acc_metric.update_state(Y_train, val_logits)
            val_acc = val_acc_metric.result()
            print("Validation acc: %.4f" % (float(val_acc),))
            val_acc_metric.reset_state()
            
        end_time = time.time()
        elapsed_time = end_time - start_time
        minutes = elapsed_time // 60
        remaining_seconds = elapsed_time % 60
        print(f"Batch time: {minutes:.2f} minutes {remaining_seconds:.2f} seconds")
        
        if i % save_epoch == 0:
            saveModel(model, checkpoint_path)
        saveEpochFile(epochFilePath, i)
        
        print("------------------------------------")
    saveModel(model, checkpoint_path)

    return model
        
def saveModel(model, checkpoint_path):
    print("Saving model... ", end='')
    model.save(checkpoint_path)
    print("Model Saved.")
            

def createElements(batch_size, height, width, multiply):
    totalBatchSize = batch_size*multiply
    X_LL = np.zeros((totalBatchSize, width*height))
    X_LH = np.zeros((totalBatchSize, width*height))
    X_HL = np.zeros((totalBatchSize, width*height))
    X_HH = np.zeros((totalBatchSize, width*height))
    X_index = np.zeros((totalBatchSize, 1))
    Y = np.zeros((totalBatchSize, 1))
    
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

def evaluate(model, listInput, posPath, negPath, height, width):
    X_LL_test, X_LH_test, X_HL_test, X_HH_test, Y_test = getBatch(listInput, posPath, negPath, 0, len(listInput) - 1, len(listInput) - 1, height, width)
    model_out = model.predict([X_LL_test,X_LH_test,X_HL_test,X_HH_test])
    passCnt = 0
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for i in range(len(Y_test)):
        if np.argmax(model_out[i, :]) == Y_test[i]:
            passCnt = passCnt + 1

        if Y_test[i] ==0:
            if np.argmax(model_out[i, :]) == Y_test[i]:
                TP = TP + 1
            else:
                FN = FN + 1
        else:
            if np.argmax(model_out[i, :]) == Y_test[i]:
                TN = TN + 1
            else:
                FP = FP + 1

    start = "\033[1m"
    end = "\033[0;0m"
    print(f'{start}confusion matrix (test / validation){end}')
    print(f'{start}true positive:  {end}{str(TP)}')
    print(f'{start}false positive: {end}{str(FP)}')
    print(f'{start}true negative:  {end}{str(TN)}')
    print(f'{start}false negative: {end}{str(FN)}')
    print('\n')
    print(f'{start}accuracy:  {end}' + "{:.4f} %".format(100 * (TP + TN) / (TP + FP + FN + TN)))
    print(f'{start}precision: {end}' + "{:.4f} %".format(100*TP/(TP + FP)))
    print(f'{start}recall:  {end}' + "{:.4f} %".format(100*TP/(TP + FN)))

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

    parser.add_argument('--batch_size', type=int, help='Batch size for epoch in training', default=32)
    parser.add_argument('--height', type=int, help='Image height resize', default=375)
    parser.add_argument('--width', type=int, help='Image width resize', default=500)
    
    parser.add_argument('--loadCheckPoint', type=str2bool, help='Enable Checkpoint Load', default='True')
    
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))