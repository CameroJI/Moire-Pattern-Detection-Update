import sys
import argparse
from PIL import Image, ImageOps
import os
import time
from os import listdir
from os.path import isfile, join
from concurrent.futures import ThreadPoolExecutor, as_completed
from haar2D import fwdHaarDWT2D

width = 0
height = 0

def main(args):
    global width, height
    origenPositiveImages = args.origenPositiveImages
    origenNegativeImages = args.origenNegativeImages
    
    outputPositiveImages = args.outputPositiveImages
    outputNegativeImages = args.outputNegativeImages
    
    width = args.width
    height = args.height
    
    createTrainingData(origenPositiveImages, origenNegativeImages, outputPositiveImages, outputNegativeImages)

def transformImageAndSave(image, f, customStr, path):
    cA, cH, cV, cD = fwdHaarDWT2D(image)

    fileName = os.path.splitext(f)[0]
    fLL = f.replace(fileName, f'{fileName}_{customStr}LL').replace(os.path.splitext(f)[-1], '.png')
    fLH = f.replace(fileName, f'{fileName}_{customStr}LH').replace(os.path.splitext(f)[-1], '.png')
    fHL = f.replace(fileName, f'{fileName}_{customStr}HL').replace(os.path.splitext(f)[-1], '.png')
    fHH = f.replace(fileName, f'{fileName}_{customStr}HH').replace(os.path.splitext(f)[-1], '.png')
    
    cA = Image.fromarray(cA)
    cH = Image.fromarray(cH)
    cV = Image.fromarray(cV)
    cD = Image.fromarray(cD)
    
    cA = cA.resize((width, height), Image.LANCZOS)
    cH = cH.resize((width, height), Image.LANCZOS)
    cV = cV.resize((width, height), Image.LANCZOS)
    cD = cD.resize((width, height), Image.LANCZOS)
    
    cA = cA.convert('L')
    cH = cH.convert('L')
    cV = cV.convert('L')
    cD = cD.convert('L')
    
    cA.save(join(path, fLL))
    cH.save(join(path, fLH))
    cV.save(join(path, fHL))
    cD.save(join(path, fHH))

def augmentAndTransformImage(f, mainFolder, trainFolder):
    global width, height

    try:
        img = PreprocessImage(join(mainFolder, f), width, height)
    except Exception:
        print(f'Error: couldn\'t read the file {f}. Make sure only images are present in the folder')
        return None

    wdChk, htChk = img.size
    if htChk > wdChk:
        img = img.rotate(-90, expand=1)
    transformImageAndSave(img, f, '', trainFolder)

    imgGray = PreprocessImage(img.rotate(45, expand=True), width, height)
    transformImageAndSave(imgGray, f, '45_', trainFolder)

    imgGray = PreprocessImage(img.rotate(90, expand=True), width, height)
    transformImageAndSave(imgGray, f, '90_', trainFolder)

    return True

def PreprocessImage(imgInput, width, height):
    if isinstance(imgInput, str):
        imgPath = imgInput
        if not isfile(imgPath):
            raise FileNotFoundError(f"La imagen en la ruta {imgPath} no fue encontrada.")
        img = Image.open(imgPath)
    elif isinstance(imgInput, Image.Image):
        img = imgInput
    else:
        raise ValueError("La entrada debe ser una ruta de archivo o una instancia de PIL.Image.Image.")
    
    w, h = img.size
    
    if w / h > width / height:  
        proportion = width / w
    else:  
        proportion = height / h

    new_width = int(w * proportion)
    new_height = int(h * proportion)

    img = img.resize((new_width, new_height), Image.LANCZOS)

    delta_w = max(0, width - new_width)
    delta_h = max(0, height - new_height)

    padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))

    img = ImageOps.expand(img, padding, fill="black")

    img = img.crop((0, 0, width, height))

    return img.convert('L')

def writelist(fileNames, outputFile):    
    with open(outputFile, 'w') as file:
        for idx, name in enumerate(fileNames):
            file.write(f"{name}\n") if idx < len(fileNames) - 1 else file.write(f"{name}")

def createTrainingData(origenPositiveImagePath, origenNegativeImagePath, outputPositiveImagePath, outputNegativeImagePath):
    # get image files by classes
    positiveImageFiles = [f for f in listdir(origenPositiveImagePath) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    negativeImageFiles = [f for f in listdir(origenNegativeImagePath) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    positiveCount = len(positiveImageFiles)
    negativeCount = len(negativeImageFiles)

    print(f'positive samples: {positiveCount}')
    print(f'negative samples: {negativeCount}')

    # create folders (not tracked by git)
    if not os.path.exists(outputPositiveImagePath):
        os.makedirs(outputPositiveImagePath)
    if not os.path.exists(outputNegativeImagePath):
        os.makedirs(outputNegativeImagePath)

    Knegative = 0
    Kpositive = 0

    start_time = time.time()

    # create positive training images using multithreading
    with ThreadPoolExecutor() as executor:
        future_to_image = {executor.submit(augmentAndTransformImage, f, origenPositiveImagePath, outputPositiveImagePath): f for f in positiveImageFiles}
        for n, future in enumerate(as_completed(future_to_image)):
            ret = future.result()
            print(f"Transformed Positive Image {n + 1}/{positiveCount}")
            if ret is not None:
                Kpositive += 3

    # create negative training images using multithreading
    with ThreadPoolExecutor() as executor:
        future_to_image = {executor.submit(augmentAndTransformImage, f, origenNegativeImagePath, outputNegativeImagePath): f for f in negativeImageFiles}
        for n, future in enumerate(as_completed(future_to_image)):
            ret = future.result()
            print(f"Transformed Negative Image {n + 1}/{negativeCount}")
            if ret is not None:
                Knegative += 3
    
    writelist(positiveImageFiles, join(outputPositiveImagePath, 'positiveFiles.lst'))
    writelist(negativeImageFiles, join(outputNegativeImagePath, 'negativeFiles.lst'))
    
    print('Total positive files after augmentation: ', Kpositive)
    print('Total negative files after augmentation: ', Knegative)

    end_time = time.time()
    elapsed_time = end_time - start_time
    hours = elapsed_time // 3600
    elapsed_time %= 3600
    minutes, seconds = divmod(elapsed_time, 60)
    print(f"\nImages Transformed in {int(hours)} hours, {int(minutes)} minutes, {seconds:.2f} seconds.")

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--origenPositiveImages', type=str, help='Directory with positive (Moiré pattern) images.')
    parser.add_argument('--origenNegativeImages', type=str, help='Directory with negative (Normal) images.')

    parser.add_argument('--outputPositiveImages', type=str, help='Directory with transformed positive Images.')
    parser.add_argument('--outputNegativeImages', type=str, help='Directory with transformed negative Images.')
    
    parser.add_argument('--width', type=int, help='width of the images.', default=1000)
    parser.add_argument('--height', type=int, help='height of the images.', default=750)
    
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))