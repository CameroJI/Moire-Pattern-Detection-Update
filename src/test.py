#Use this file for evaluating on a dataset that is not used for training

from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
import sys
import argparse
import time
from math import ceil
from os import listdir
from os.path import join, exists
from PIL import Image
import io
from mCNN import createModel
from haar2D import fwdHaarDWT2D
from train import createElements, defineEpochRange, scaleData
import random
from keras.utils import to_categorical # type: ignore
import tensorflow as tf
    
test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.BinaryAccuracy(name='test_accuracy')

def load_model(model_path):
    model_extension = Path(model_path).suffix.lower()

    if model_extension in ['.h5', '.keras']:
        return tf.keras.models.load_model(model_path)
    
    elif model_extension == '.tflite':
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter
    
    else:
        raise ValueError(f"Unsupported model file format: {model_extension}")
    
def main(args):
    weights_file = (args.weightsFile)
    positiveImagePath = (args.positiveTestImages)
    negativeImagePath = (args.negativeTestImages)
    batch_size = (args.batch_size)
    height = (args.height)
    width = (args.width)
    
    datasetList, numClasses = createIndex(positiveImagePath, negativeImagePath)
    
    if exists(weights_file):
        print("El archivo existe")
        CNN_model = load_model(weights_file)
        evaluateFolder(CNN_model, datasetList, positiveImagePath, negativeImagePath, batch_size, numClasses, height, width)
    else:
        print("El archivo del modelo no existe en el directorio establecido.")
    
def test_step(model, X_LL_test, X_LH_test, X_HL_test, X_HH_test, labels):
    if isinstance(model, tf.keras.Model):
        predictions = model([X_LL_test, X_LH_test, X_HL_test, X_HH_test], training=False)
        
    elif isinstance(model, tf.lite.Interpreter):
        # Obtener detalles de las entradas y salidas
        input_details = model.get_input_details()
        output_details = model.get_output_details()
        
        # Número de muestras
        N = X_LL_test.shape[0]

        # Almacenar resultados
        results = []

        for i in range(N):
            sample_LL = np.array([X_LL_test[i]]).astype(np.float32)
            sample_LH = np.array([X_LH_test[i]]).astype(np.float32)
            sample_HL = np.array([X_HL_test[i]]).astype(np.float32)
            sample_HH = np.array([X_HH_test[i]]).astype(np.float32)
                        
            # Configurar las entradas del intérprete TensorFlow Lite
            model.set_tensor(input_details[0]['index'], sample_LL)
            model.set_tensor(input_details[1]['index'], sample_LH)
            model.set_tensor(input_details[2]['index'], sample_HL)
            model.set_tensor(input_details[3]['index'], sample_HH)

            # Ejecutar el intérprete
            model.invoke()

            # Obtener los resultados del intérprete
            output_data = model.get_tensor(output_details[0]['index'])
            results.append(output_data)

        # Convertir resultados a un array de numpy
        predictions = np.squeeze(np.array(results), axis=1)
    else:
        raise TypeError("Unsupported model type")

    t_loss = tf.keras.losses.binary_crossentropy(labels, predictions)
    test_loss(t_loss)
    test_accuracy(labels, predictions)

    return test_loss.result(), test_accuracy.result()
    
def evaluateFolder(model, listInput, posPath, negPath, batch_size, numClasses, height, width):
    n = len(listInput)
    total_loss = 0
    total_accuracy = 0
    steps = ceil(n/batch_size)
    print()
    
    test_loss.reset_state()
    test_accuracy.reset_state()
    
    start_time_full = time.time()
    for i in range(steps):
        start_time = time.time()
        
        start, end = defineEpochRange(i, batch_size, n)
        
        X_LL_test, X_LH_test, X_HL_test, X_HH_test, Y_test = getEvaluationBatch(listInput, posPath, negPath, start, end, batch_size, height, width)

        data = [X_LL_test, X_LH_test, X_HL_test, X_HH_test]
        labels = to_categorical(Y_test, numClasses)
        
        data = np.array(data)
        labels = np.array(labels)
        
        loss, accuracy = test_step(model, X_LL_test, X_LH_test, X_HL_test, X_HH_test, labels)
        if i==0:    print()
        print(f"Testing {end - start} images ({i + 1}/{steps})", end='\t')
        print(f'start: {start}\tend: {end}\tTotal Images:{len(listInput)}')
        print(f'Test Loss: {loss*100:.2f}%\t Test Accuracy: {accuracy*100:.2f}%\n')
        
        total_loss += loss
        total_accuracy += accuracy
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        minutes = elapsed_time // 60
        remaining_seconds = elapsed_time % 60
        print(f"Batch time: {int(minutes)} minutes {remaining_seconds:.2f} seconds")
        print("------------------------------------")

    # Iterar sobre el dataset y calcular la precisión y la pérdida
    print(f'\nTotal Loss: {total_loss}')
    print(f'Total Accuracy: {total_accuracy}')

    mean_loss = total_loss / steps
    mean_accuracy = total_accuracy / steps   

    print(f'\nMean Loss: {mean_loss*100:2f}%')
    print(f'Mean Accuracy: {mean_accuracy*100:2f}%')
    
    end_time_full = time.time()
    elapsed_time = end_time_full - start_time_full
    hours = elapsed_time // 3600
    elapsed_time %= 3600
    minutes, seconds = divmod(elapsed_time, 60)
    print(f"\nTotal testing time: {int(hours)} hours, {int(minutes)} minutes, {seconds:.2f} seconds.\n\n")     

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
    
    