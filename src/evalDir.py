from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
import sys
import argparse
import time
from math import ceil
from os import listdir, makedirs, remove
from os.path import join, exists
from PIL import Image
import io
from haar2D import fwdHaarDWT2D
from train import createElements, defineEpochRange, scaleData
import tensorflow as tf

def load_model(model_path):
    model_extension = Path(model_path).suffix.lower()

    if model_extension in ['.h5', '.keras']:
        model = tf.keras.models.load_model(model_path, compile=False)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        print(f"Modelo {model_extension} encontrado y cargado correctamente!\n")
        return model
    
    elif model_extension == '.tflite':
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        print("Modelo .tflite encontrado y cargado correctamente!\n")
        return interpreter
        
    else:
        raise ValueError(f"Unsupported model file format: {model_extension}")
    
def main(args):
    weights_file = args.weightsFile
    dirImages = args.dirImages
    batch_size = args.batch_size
    height = args.height
    width = args.width
    filePredictionDir = args.filePredictionDir
    
    if not exists(filePredictionDir):
        makedirs(filePredictionDir)
    
    datasetList = createIndex(dirImages)
    
    if exists(weights_file):
        CNN_model = load_model(weights_file)
        predictionMethod(CNN_model, datasetList, dirImages, batch_size, height, width, filePredictionDir)
    else:
        print("El archivo del modelo no existe en el directorio establecido.")
    
def test_step(model, X_LL_test, X_LH_test, X_HL_test, X_HH_test, filePredictionDir, dataset, path):
    if isinstance(model, tf.keras.Model):
        predictions = model([X_LL_test, X_LH_test, X_HL_test, X_HH_test], training=False)
        
    elif isinstance(model, tf.lite.Interpreter):
        input_details = model.get_input_details()
        output_details = model.get_output_details()
        
        N = X_LL_test.shape[0]
        results = []

        for i in range(N):
            sample_LL = np.array([X_LL_test[i]]).astype(np.float32)
            sample_LH = np.array([X_LH_test[i]]).astype(np.float32)
            sample_HL = np.array([X_HL_test[i]]).astype(np.float32)
            sample_HH = np.array([X_HH_test[i]]).astype(np.float32)
                        
            model.set_tensor(input_details[0]['index'], sample_LL)
            model.set_tensor(input_details[1]['index'], sample_LH)
            model.set_tensor(input_details[2]['index'], sample_HL)
            model.set_tensor(input_details[3]['index'], sample_HH)

            model.invoke()
            output_data = model.get_tensor(output_details[0]['index'])
            results.append(output_data)

        predictions = np.squeeze(np.array(results), axis=1)
    else:
        raise TypeError("Unsupported model type")
    
    confidences = predictions.numpy()
        
    # Guardar imágenes incorrectas
    positive_incertudumbre = []
    negative_incertudumbre = []
    positive_predictions = []
    negative_predictions = []

    for idx, confidence in enumerate(confidences):
        image_name = dataset[idx]
        image_path = join((path.split('/')[-1]).replace('_clonadas',''), image_name.split("_")[0])
        
        if confidence[1] > confidence[0]:  # True Positive
            if confidence[1] >= 0.9:
                positive_predictions.append(image_path)
            else:
                positive_incertudumbre.append(image_path)
        else:  # True Negative
            if confidence[0] >= 0.9:
                negative_predictions.append(image_path)
            else:
                negative_incertudumbre.append(image_path)
                
    print(f'Predicciones positivas : {len(positive_predictions)}\nPredicciones negativas : {len(negative_predictions)}\n')
    print(f'Predicciones positivas con incertidumbre: {len(positive_incertudumbre)}\nPredicciones negativas con incertidumbre: {len(negative_incertudumbre)}\n')

    # Save filenames to text files  
    writeImageInFile(filePredictionDir, 'Real_predictions.txt', positive_predictions)
    writeImageInFile(filePredictionDir, 'Ataque_predictions.txt', negative_predictions)
    writeImageInFile(filePredictionDir, 'Real_incertudumbre.txt', positive_incertudumbre)
    writeImageInFile(filePredictionDir, 'Ataque_incertudumbre.txt', negative_incertudumbre)
            
def writeImageInFile(root, file, listNames):
    with open(join(root, file), 'a') as f:
        for name in listNames:
            f.write(f"{name}\n")
    
    
def predictionMethod(model, listInput, path, batch_size, height, width, filePredictionDir):
    n = len(listInput)
    steps = ceil(n/batch_size)
    print()
    start_time_full = time.time()
    for i in range(steps):
        start_time = time.time()
        
        start, end = defineEpochRange(i, batch_size, n)
        
        X_LL_test, X_LH_test, X_HL_test, X_HH_test = getEvaluationBatch(listInput, path, start, end, batch_size, height, width)

        data = [X_LL_test, X_LH_test, X_HL_test, X_HH_test]
        
        data = np.array(data)
        
        test_step(model, X_LL_test, X_LH_test, X_HL_test, X_HH_test, filePredictionDir, listInput[start:end], path)
        if i==0:    print()
        print(f"Predicting {end - start} images ({i + 1}/{steps})", end='\t')
        print(f'start: {start}\tend: {end}\tTotal Images:{len(listInput)}')
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        minutes = elapsed_time // 60
        remaining_seconds = elapsed_time % 60
        print(f"Batch time: {int(minutes)} minutes {remaining_seconds:.2f} seconds")
        print("------------------------------------")
    
    end_time_full = time.time()
    elapsed_time = end_time_full - start_time_full
    hours = elapsed_time // 3600
    elapsed_time %= 3600
    minutes, seconds = divmod(elapsed_time, 60)
    print(f"\nTotal prediction time: {int(hours)} hours, {int(minutes)} minutes, {seconds:.2f} seconds.\n\n")     

def createIndex(path):
    imgList = list(listdir(path))
    
    return imgList

def getEvaluationBatch(listInput, path, start, end, batch_size, height, width):
    X_LL, X_LH, X_HL, X_HH, X_index, Y, totalBatchSize = createElements(end-start, height, width, 1)

    for sampleIndex, f in enumerate(listInput[start:end]):
        cA, cH, cV, cD = readAndBuildBatch(f, path, height, width)
        X_LL, X_LH, X_HL, X_HH, X_index = transformImage(cA, cH, cV, cD, X_LL, X_LH, X_HL, X_HH, X_index, sampleIndex, height, width)
        
    X_LL = X_LL.reshape((totalBatchSize, height, width, 1))
    X_LH = X_LH.reshape((totalBatchSize, height, width, 1))
    X_HL = X_HL.reshape((totalBatchSize, height, width, 1))
    X_HH = X_HH.reshape((totalBatchSize, height, width, 1))

    return X_LL, X_LH, X_HL, X_HH

def transformImage(imgLL, imgLH, imgHL, imgHH, X_LL, X_LH, X_HL, X_HH, X_index, sampleIndex, height, width):    
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
    
    X_index[sampleIndex, 0] = sampleIndex
    
    return X_LL, X_LH, X_HL, X_HH, X_index

def readAndBuildBatch(f, path, height, width):
    try:
        cA, cH, cV, cD = imageTransformation(path, f, height, width)
    except Exception:
        print(f'No se pudo leer el archivo {f} en la carpeta {path}, por lo que se eliminará.')
        remove(join(path, f))
    return cA, cH, cV, cD
    

def imageTransformation(imageRoot, f, height, width):
    try:
        img = Image.open(join(imageRoot, f))
    except Exception:
        print(f'Error: couldn\'t read the file {f}. Make sure only images are present in the folder')
        return None

    w, h = img.size
    img = img.resize((height, width)) if h > w else img.resize((width, height))
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
    
    parser.add_argument('--dirImages', type=str, help='Directory with images.')
    
    parser.add_argument('--batch_size', type=int, help='Batch size for epoch in training', default=32)
    parser.add_argument('--height', type=int, help='Image height resize', default=375)
    parser.add_argument('--width', type=int, help='Image width resize', default=500)
    
    parser.add_argument('--filePredictionDir', type=str, help='Directory to save predictions.', default='./predicciones')
    
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))