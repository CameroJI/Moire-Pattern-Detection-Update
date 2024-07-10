#Use this file for evaluating on a dataset that is not used for training

from matplotlib import pyplot as plt
import numpy as np
import sys
import argparse
from math import ceil
from os import listdir
from os.path import isfile, join, splitext
from PIL import Image
from sklearn import preprocessing
import io
from sklearn.model_selection import train_test_split
from mCNN import createModel
import createTrainingData
from matplotlib.pyplot import imshow
from haar2D import fwdHaarDWT2D
from train import evaluate, createElements, defineEpochRange, scaleData
import random
from keras.utils import to_categorical # type: ignore
import tensorflow as tf
    
def main(args):
    weights_file = (args.weightsFile)
    positiveImagePath = (args.positiveTestImages)
    negativeImagePath = (args.negativeTestImages)
    batch_size = (args.batch_size)
    height = (args.height)
    width = (args.width)
    
    datasetList, numClasses = createIndex(positiveImagePath, negativeImagePath)
    
    CNN_model = createModel(height, width, 1, numClasses)
    CNN_model.compile(loss='categorical_crossentropy', # using the cross-entropy loss function
        optimizer='adam', # using the Adam optimizer
        metrics=['accuracy']) # reporting the accuracy 
    CNN_model.load_weights(weights_file)
    
    evaluateFolder(CNN_model, datasetList, positiveImagePath, negativeImagePath, batch_size, numClasses, height, width)
    #evaluate(CNN_model,X_LL,X_LH,X_HL,X_HH, Y)
    
def evaluateFolder(model, listInput, posPath, negPath, batch_size, numClasses, height, width):
    n = len(listInput)
    total_loss = 0
    total_accuracy = 0
    steps = ceil(n/batch_size)
    print()
    
    for i in range(steps):
        start, end = defineEpochRange(i, batch_size, n)
        print(f"Training {end - start} images.", end='\t')
        print(f'start: {start}\tend: {end}\tn:{len(listInput)}')
        
        X_LL_test, X_LH_test, X_HL_test, X_HH_test, Y_test = getEvaluationBatch(listInput, posPath, negPath, start, end, batch_size, height, width)
        #Y_test = to_categorical(Y_test, numClasses)
        batchTemp = end - start
        data = [X_LL_test, X_LH_test, X_HL_test, X_HH_test]
        labels = to_categorical(Y_test, numClasses)
        
        data = np.array(data)
        labels = np.array(labels)
        
        def data_generator():
            for i in range(batchTemp):
                # Para cada ejemplo, devolver un diccionario con los nombres de entrada esperados por el modelo
                yield ({
                    'input_1': data[0][i],
                    'input_2': data[1][i],
                    'input_3': data[2][i],
                    'input_4': data[3][i]
                }, labels[i])

        # Especificación de tipo para el dataset
        dataset = tf.data.Dataset.from_generator(data_generator,
                                        output_signature=(
                                            {
                                                'input_1': tf.TensorSpec(shape=(375, 500, 1), dtype=tf.float32),
                                                'input_2': tf.TensorSpec(shape=(375, 500, 1), dtype=tf.float32),
                                                'input_3': tf.TensorSpec(shape=(375, 500, 1), dtype=tf.float32),
                                                'input_4': tf.TensorSpec(shape=(375, 500, 1), dtype=tf.float32)
                                            },
                                            tf.TensorSpec(shape=(2,), dtype=tf.int32)))

        dataset = dataset.shuffle(buffer_size=batchTemp).batch(batchTemp)
        
        iterator = iter(dataset)
        try:
            while True:
                batch = next(iterator)
                x_batch = batch[0]  # Obtener el diccionario de entradas X
                y_batch = batch[1]
        
                loss, accuracy = model.test_on_batch(x_batch, y_batch)
                print(f'Batch Loss: {loss*100}%')
                print(f'Batch Accuracy: {accuracy*100}%\n') 
                total_loss += loss
                total_accuracy += accuracy
                # Procesar el lote aquí
        except StopIteration:
            continue

    # Iterar sobre el dataset y calcular la precisión y la pérdida
    print(f'\nTotal Loss: {total_loss}')
    print(f'Total Accuracy: {total_accuracy}')

    mean_loss = total_loss / steps
    mean_accuracy = total_accuracy / steps   

    print(f'\nMean Loss: {mean_loss*100}%')
    print(f'Mean Accuracy: {mean_accuracy*100}%')      

def createIndex(posPath, negPath):
    posList = list(listdir(posPath))
    negList = list(listdir(negPath))

    datasetList = [(i, 1) for i in posList]
    datasetList.extend((i, 0) for i in negList)

    #dataset split for training and evaluation
    random.shuffle(datasetList)
    
    classes = len(np.unique(np.array([fila[1] for fila in datasetList])))
    
    return datasetList, classes

def getEvaluationBatch(listInput, posPath, negPath, start, end, batch_size, height, width):
    X_LL, X_LH, X_HL, X_HH, X_index, Y, totalBatchSize = createElements(end-start, height, width, 1)

    for sampleIndex, f in enumerate(listInput[start:end]):
        cA, cH, cV, cD = readAndBuildBatch(f, posPath, negPath, height, width)
        X_LL, X_LH, X_HL, X_HH, X_index, Y = transformImage(cA, cH, cV, cD, X_LL, X_LH, X_HL, X_HH, X_index, Y, sampleIndex, int(f[1]), height, width)
        
    X_LL = X_LL.reshape((totalBatchSize, height, width, 1))
    X_LH = X_LH.reshape((totalBatchSize, height, width, 1))
    X_HL = X_HL.reshape((totalBatchSize, height, width, 1))
    X_HH = X_HH.reshape((totalBatchSize, height, width, 1))

    return X_LL, X_LH, X_HL, X_HH, Y

def transformImage(imgLL, imgLH, imgHL, imgHH, X_LL, X_LH, X_HL, X_HH, X_index, Y, sampleIndex, y, height, width):    
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
    
    Y[sampleIndex, 0] = y
    X_index[sampleIndex, 0] = sampleIndex
    
    return X_LL, X_LH, X_HL, X_HH, X_index, Y

def readAndBuildBatch(f, posPath, negPath, height, width):
    file = str(f[0])
    y = int(f[1])
    path = posPath if y == 1 else negPath
    
    cA, cH, cV, cD = imageTransformation(path, file, height, width)
    
    return cA, cH, cV, cD
    

def imageTransformation(imageRoot, f, height, width):
    try:
        img = Image.open(join(imageRoot, f))
    except Exception:
        print(f'Error: couldn\'t read the file {f}. Make sure only images are present in the folder')
        return None

    w, h = img.size
    img = img.resize((750, 1000)) if h > w else img.resize((1000, 750))
    imgGray = img.convert('L')
    wdChk, htChk = imgGray.size
    if htChk > wdChk:
        imgGray = imgGray.rotate(-90, expand=1)
        
    cA, cH, cV, cD  = fwdHaarDWT2D(imgGray)
    
    cA = Image.fromarray(cA)
    cH = Image.fromarray(cH)
    cV = Image.fromarray(cV)
    cD = Image.fromarray(cD)
    
    cA = cA.resize((width, height))
    cH = cH.resize((width, height))
    cV = cV.resize((width, height))
    cD = cD.resize((width, height))
    
    cA = Image.open(getTiffFromJpg(cA))
    cH = Image.open(getTiffFromJpg(cH))
    cV = Image.open(getTiffFromJpg(cV))
    cD = Image.open(getTiffFromJpg(cD))
            
    return cA, cH, cV, cD

def getTiffFromJpg(img):
    tiff_bytes_io = io.BytesIO()
    
    img.save(tiff_bytes_io, format="TIFF")
    tiff_bytes_io.seek(0)
    
    return tiff_bytes_io

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--weightsFile', type=str, help='Saved CNN model file', default='./checkpoint/cp.h5')
    
    parser.add_argument('--positiveTestImages', type=str, help='Directory with positive (Moiré pattern) images.')
    parser.add_argument('--negativeTestImages', type=str, help='Directory with negative (Normal) images.')
    
    parser.add_argument('--batch_size', type=int, help='Batch size for epoch in training', default=32)
    parser.add_argument('--height', type=int, help='Image height resize', default=375)
    parser.add_argument('--width', type=int, help='Image width resize', default=500)
    
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
    
    