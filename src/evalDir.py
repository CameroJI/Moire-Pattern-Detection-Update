from pathlib import Path
import sys
import argparse
import tensorflow as tf
from os import listdir
from os.path import join, basename
import numpy as np
from math import pi
from skimage.filters import gabor
from sklearn import preprocessing
from PIL import Image
import pywt
import pywt.data
import io
import json

def main(args):
    weights_file = args.weightsFile
    rootPath = args.rootPath
    height = args.height
    width = args.width
    
    model = load_model(weights_file)
    
    evaluateFolders(model, rootPath, height, width)
            
def evaluateFolders(model, root, height, width):
    try:
        i = 0
        for idx, file in enumerate(listdir(root)):
            img = Image.open(join(root, file))
            
            jsonWrite = join(root, f'{file}_jsonPredict.son')
            
            X_LL, X_LH, X_HL, X_HH, Y = getEvaluationBatch(img, height, width)
            score, ocurrences, prediction = evaluate(model, X_LL, X_LH, X_HL, X_HH, Y)
            # createJson(jsonWrite, basename(root), basename(dirPath), first, score, ocurrences, prediction)
            if prediction == 'WARNING' or prediction == 'FAIL':
                i+=1
            print('---------------------------------------------------------------------')
            print(f'{file}',end='\t')
            print(f'Score: {score}\tOcurrences: {ocurrences}\tPrediction: {prediction}\t\n')
    except:
        print(f'Archivo: {file} no se pudo procesar.')
        
    print(f'Total de Ataques detectados: {i}/{len(listdir(root))}')
    
def createJson(path, basename, dirPath, n, score, ocurrences, prediction):
    results = {
        "root": {
            "rootFolder": basename,
            "dir": dirPath,
            "imageNumber":  n
        },
        "results": {
            "score": score,
            "ocurrences": ocurrences,
            "prediction": prediction
        }
    }
    print(results['results'])
    
    with open(path, 'w') as json_file:
        json.dump(results, json_file, indent=4) 
        
def evaluate(model, X_LL_test,X_LH_test,X_HL_test,X_HH_test,y_test):
    model_out = model([X_LL_test, X_LH_test, X_HL_test, X_HH_test], training=False)
    TP = 0
    TN = 0

    for i in range(len(y_test)):
        if np.argmax(model_out[i, :]) == y_test[i]:
            TP += 1
        else:
            TN += 1
            
    if TN == 0:
        str_label = 'PASS'
        
    elif TN > 0 and TN < 2:
        str_label = 'WARNING'
        
    else:
        str_label = 'FAIL'
    
    ocurrences = TN
    precision = round(ocurrences/3, 3)

    return precision, ocurrences, str_label
        
def getEvaluationBatch(img, height, width):
    X_LL, X_LH, X_HL, X_HH, X_index, Y, totalBatchSize = createElements(1, height, width, 3)

    w, h = img.size
    if h > w:
        img = img.resize((height, width))
    else:
        img = img.resize((width, height))

    imgGray1 = img.convert('L')
    wdChk, htChk = imgGray1.size
    if htChk > wdChk:
        imgGray1 = imgGray1.rotate(-90, expand=1)

    imgGray2 = imgGray1.transpose(Image.ROTATE_180)

    imgGray3 = imgGray1.transpose(Image.FLIP_LEFT_RIGHT)
    
    images = [imgGray1, imgGray2, imgGray3]
    
    for sampleIndex, img in enumerate(images):
        
        cA, cH, cV, cD = imageTransformation(img, height, width)
        X_LL, X_LH, X_HL, X_HH, X_index = transformImage(cA, cH, cV, cD, X_LL, X_LH, X_HL, X_HH, X_index, sampleIndex, height, width)
        
    X_LL = X_LL.reshape((totalBatchSize, height, width, 1))
    X_LH = X_LH.reshape((totalBatchSize, height, width, 1))
    X_HL = X_HL.reshape((totalBatchSize, height, width, 1))
    X_HH = X_HH.reshape((totalBatchSize, height, width, 1))

    return X_LL, X_LH, X_HL, X_HH, Y

def scaleData(inp, minimum, maximum):
    minMaxScaler = preprocessing.MinMaxScaler(copy=True, feature_range=(minimum,maximum))
    inp = inp.reshape(-1, 1)
    inp = minMaxScaler.fit_transform(inp)
    
    return inp

def fwdHaarDWT2D(img):
    coeffs2 = pywt.dwt2(img, 'bior1.3')
    LL, (HL, LH, HH) = coeffs2
    gaborImg = gaborFilter(img)
    
    return LL, LH, HL, HH

def gaborFilter(img, frequency=0.56, theta=pi/2):
    filt_real, filt_imag = gabor(np.array(img, dtype=np.float32), frequency=frequency, theta=theta)
    
    magnitude = np.sqrt(filt_real**2 + filt_imag**2)
    magnitude = (magnitude - np.min(magnitude)) / (np.max(magnitude) - np.min(magnitude)) * 255
    
    return magnitude

def createElements(batch_size, height, width, multiply):
    totalBatchSize = batch_size*multiply
    X_LL = np.zeros((totalBatchSize, width*height))
    X_LH = np.zeros((totalBatchSize, width*height))
    X_HL = np.zeros((totalBatchSize, width*height))
    X_HH = np.zeros((totalBatchSize, width*height))
    X_index = np.zeros((totalBatchSize, 1))
    Y = np.ones((totalBatchSize, 1))
    
    return X_LL, X_LH, X_HL, X_HH, X_index, Y, totalBatchSize

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

def imageTransformation(img, height, width):
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
    
def load_model(model_path):
    model_extension = Path(model_path).suffix.lower()

    if model_extension in ['.h5', '.keras']:
        model = tf.keras.models.load_model(model_path, compile=False)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        print(f"Modelo {model_extension} encontrado y cargado correctamente!\n")
        return model
    
    else:
        raise ValueError(f"Unsupported model file format: {model_extension}")

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--weightsFile', type=str, help='Saved CNN model file', default='./checkpoint/cp.h5')
    
    parser.add_argument('--rootPath', type=str, help='Directory with (Moiré pattern) images.', default='./')
    
    parser.add_argument('--height', type=int, help='Image height resize', default=800)
    parser.add_argument('--width', type=int, help='Image width resize', default=1400)
        
    return parser.parse_args(argv)
    
if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))