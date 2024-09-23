from readJson import getScoresFromJSON
from pathlib import Path
import sys
import argparse
import tensorflow as tf
from os import listdir
from os.path import join, basename, isdir
import numpy as np
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
    for idx, dir in enumerate(listdir(root)):
        dirPath = join(root, dir)
        
        if isdir(dirPath):
            fileNames = [file for file in listdir(dirPath) if '_front_large.jpg' in file]
            
            if len(fileNames) >= 1:
                numArray = ([int(''.join([char for char in name if char.isdigit()])) for name in fileNames])
                
                first = min(numArray)
                last = max(numArray)
                
                firstImg = f'{first}_front_large.jpg'
                lastImg = f'{last}_front_large.jpg'
                
                firtJson = f'{first}_front_result.json'
                lastJson = f'{last}_front_result.json'
                        
                firstImg = PreprocessImage(join(dirPath, firstImg), width, height)
                lastImg = PreprocessImage(join(dirPath, lastImg), width, height)
                
                firstJsonWrite = join(dirPath, f'{first}_jsonPredict.json')
                lastJsonWrite = join(dirPath, f'{last}_jsonPredict.json')
                
                X_LL, X_LH, X_HL, X_HH, Y = getEvaluationBatch(firstImg, height, width)
                score, ocurrences, prediction = evaluate(model, X_LL, X_LH, X_HL, X_HH, Y)
                createJson(firstJsonWrite, basename(root), basename(dirPath), first, score, ocurrences, prediction)
                
                X_LL, X_LH, X_HL, X_HH, Y = getEvaluationBatch(lastImg, height, width)
                score, ocurrences, prediction = evaluate(model, X_LL, X_LH, X_HL, X_HH, Y)
                createJson(lastJsonWrite, basename(root), basename(dirPath), last, score, ocurrences, prediction)
                
                print(f'\n{idx+1}/{len(listdir(root))}\n')
                
def elementsFromJson(jsonObject):
    scores = {'result_score': 0.0,'result_attack': 0.0, 'result_ocurrences': 0.0}
    #scores = {'result_attack': 0.0, 'result_ocurrences': 0.0}
    
    for result in scores:
        if result in jsonObject.columns:
            resultVal = jsonObject[result].iloc[0]
        else:
            resultVal = None
        
        scores[result] = resultVal
        
    return scores
    
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
    print(f"Archivo: {join(basename, dirPath, f'{n}_front_large.jpg')}")
    print(f"Resultados: {results['results']}")
    print(f'Guardado en {path}')
    print('-' * 70)
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
        
    elif TN == 1:
        str_label = 'WARNING'
        
    else:
        str_label = 'FAIL'
    
    ocurrences = TN
    precision = round(ocurrences/3, 3)

    return precision, ocurrences, str_label
    
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
    
    parser.add_argument('--weightsFile', type=str,required=True ,help='Saved CNN model file')
    
    parser.add_argument('--rootPath',required=True, type=str, help='Directory with (Moiré pattern) images.')
    
    parser.add_argument('--height', type=int, help='Image height resize', default=800)
    parser.add_argument('--width', type=int, help='Image width resize', default=1400)
        
    return parser.parse_args(argv)
    
if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))