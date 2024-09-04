from pathlib import Path
import sys
import argparse
import tensorflow as tf
from os import listdir
from os.path import join, basename
from math import pi
from utils import waveletFunction
import numpy as np
from sklearn import preprocessing
from PIL import Image
import io

def main(args):
    weights_file = args.weightsFile
    rootPath = args.rootPath
    height = args.height
    width = args.width
    
    model = load_model(weights_file)
    
    evaluateFolders(model, rootPath, height, width)
            
def evaluateFolders(model, root, height, width):
    i = 0
    warningsCnt = 0
    failCnt = 0
    imgTotal = len(listdir(root))
    for idx, imgPath in enumerate(listdir(root)):
        try:
            path = join(root, imgPath)
            
            X_LL, X_LH, X_HL, X_HH, Y = getEvaluationBatch(path, height, width)
            score, ocurrences, prediction = evaluate(model, X_LL, X_LH, X_HL, X_HH, Y)

            if prediction == 'WARNING':
                warningsCnt+=1
            if prediction == 'FAIL':
                failCnt+=1
            if prediction == 'WARNING' or prediction == 'FAIL':
                i+=1
            print('-'*70)
            print(f'{imgPath}',end='\t')
            print(f'Score: {score}\tOcurrences: {ocurrences}\tPrediction: {prediction}\t{idx+1}/{len(listdir(root))}\n')
        
        except:
            print(f'Archivo: {imgPath} no se pudo procesar.')
            imgTotal -= 1
        
    print(f'Total de Ataques detectados: {failCnt}/{imgTotal}')
    print(f'Total de Warning detectados: {warningsCnt}/{imgTotal}')
    
    print(f'Total detectados: {i}/{imgTotal}')    
    
        
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
        
    elif TN == 1:
        str_label = 'WARNING'
        
    else:
        str_label = 'FAIL'
    
    ocurrences = TN
    precision = round(ocurrences/3, 3)

    return precision, ocurrences, str_label
        
def getEvaluationBatch(imgPath, height, width):
    X_LL, X_LH, X_HL, X_HH, X_index, Y, totalBatchSize = createElements(1, height, width, 3)

    img = PreprocessImage(imgPath, height, width)

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
    
    else:
        proportion = max(width / w, height / h)
        new_width = int(w * proportion)
        new_height = int(h * proportion)
        
        img = img.resize((new_width, new_height), Image.LANCZOS)
        
        left = (new_width - width) / 2
        top = (new_height - height) / 2
        right = (new_width + width) / 2
        bottom = (new_height + height) / 2
        
        img = img.crop((left, top, right, bottom))
    
    return img

def scaleData(inp, minimum, maximum):
    minMaxScaler = preprocessing.MinMaxScaler(copy=True, feature_range=(minimum,maximum))
    inp = inp.reshape(-1, 1)
    inp = minMaxScaler.fit_transform(inp)
    
    return inp

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
    
    imgLL = evalAugmentation(imgLL)
    imgLH = evalAugmentation(imgLH)
    imgHL = evalAugmentation(imgHL)
    imgHH = evalAugmentation(imgHH)
     
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
        
    cA, cH, cV, cD  = waveletFunction(imgGray)
    
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

def evalAugmentation(img):
    image_np = np.array(img)
    
    if len(image_np.shape) != 2:
        raise ValueError("The input image must be a 2-D grayscale image.")
    
    image_tf = tf.convert_to_tensor(image_np, dtype=tf.float32)
    image_tf = tf.expand_dims(image_tf, axis=-1)
    
    image_tf = tf.squeeze(image_tf, axis=-1)
    image_tf = tf.clip_by_value(image_tf, 0, 255)
    image_tf = tf.cast(image_tf, dtype=tf.uint8)
    
    image_prepared = Image.fromarray(image_tf.numpy())
    
    return image_prepared

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