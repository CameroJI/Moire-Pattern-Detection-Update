import sys
import argparse
from PIL import Image
from PIL import ImageOps
import sys
import os
import time
from os import listdir
from os.path import isfile, join
from PIL import Image
from haar2D import fwdHaarDWT2D

def main(args):
    origenPositiveImages = (args.origenPositiveImages)
    origenNegativeImages = (args.origenNegativeImages)
    
    outputPositiveImages = (args.outputPositiveImages)
    outputNegativeImages = (args.outputNegativeImages)
    
    createTrainingData(origenPositiveImages, origenNegativeImages, outputPositiveImages, outputNegativeImages)

    
#The wavelet decomposed images are the transformed images representing the spatial and the frequency information of the image. These images are stored as 'tiff' in the disk, to preserve that information. Each image is transformed with 180 degrees rotation and as well flipped, as part of data augmentation.

def transformImageAndSave(image, f, customStr, path):
    cA, cH, cV, cD  = fwdHaarDWT2D(image)

    fileName = (os.path.splitext(f)[0])
    fLL = f.replace(fileName, f'{fileName}_{customStr}LL').replace(os.path.splitext(f)[-1], '.tiff')
    fLH = f.replace(fileName, f'{fileName}_{customStr}LH').replace(os.path.splitext(f)[-1], '.tiff')
    fHL = f.replace(fileName, f'{fileName}_{customStr}HL').replace(os.path.splitext(f)[-1], '.tiff')
    fHH = f.replace(fileName, f'{fileName}_{customStr}HH').replace(os.path.splitext(f)[-1], '.tiff')
    cA = Image.fromarray(cA)
    cH = Image.fromarray(cH)
    cV = Image.fromarray(cV)
    cD = Image.fromarray(cD)
    cA.save(join(path, fLL))
    cH.save(join(path, fLH))
    cV.save(join(path, fHL))
    cD.save(join(path, fHH))
    
    
def augmentAndTransformImage(f, mainFolder, trainFolder):
    try:
        img = Image.open(join(mainFolder, f))
    except Exception:
        print(f'Error: couldn\'t read the file {f}. Make sure only images are present in the folder')
        return None

    w, h = img.size
    img = img.resize((750, 1000)) if h > w else img.resize((1000, 750))
    imgGray = img.convert('L')
    wdChk, htChk = imgGray.size
    if htChk > wdChk:
        imgGray = imgGray.rotate(-90, expand=1)
    transformImageAndSave(imgGray, f, '', trainFolder)

    imgGray = imgGray.transpose(Image.ROTATE_180)
    transformImageAndSave(imgGray, f, '180_', trainFolder)

    imgGray = imgGray.transpose(Image.FLIP_LEFT_RIGHT)
    transformImageAndSave(imgGray, f, '180_FLIP_', trainFolder)

    return True
    
    
def createTrainingData(origenPositiveImagePath, origenNegativeImagePath, outputPositiveImagePath, outputNegativeImagePath):
    
    # get image files by classes
    positiveImageFiles = [f for f in listdir(origenPositiveImagePath) if (isfile(join(origenPositiveImagePath, f)))]
    negativeImageFiles = [f for f in listdir(origenNegativeImagePath) if (isfile(join(origenNegativeImagePath, f)))]

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
    # create positive training images
    n = 0
    for f in positiveImageFiles:
        ret = augmentAndTransformImage(f, origenPositiveImagePath, outputPositiveImagePath)
        print(f"Transformed Positive Image {n + 1}/{positiveCount}")
        if ret is None:
            continue
        Kpositive += 3
        n += 1
        
    end_time = time.time()
    elapsed_time = end_time - start_time
    minutes = elapsed_time // 60
    remaining_seconds = elapsed_time % 60
    print(f"\nPositive Images Trained Completed in {minutes:.2f} minutes {remaining_seconds:.2f} seconds")
    print("------------------------------------\n")

    # create negative training images
    start_time = time.time()
    n = 0
    for f in negativeImageFiles:
        ret = augmentAndTransformImage(f, origenNegativeImagePath, outputNegativeImagePath)
        print(f"Transformed Negative Image {n + 1}/{negativeCount}")
        if ret is None:
            continue
        Knegative += 3
        n += 1
        
    end_time = time.time()
    elapsed_time = end_time - start_time
    minutes = elapsed_time // 60
    remaining_seconds = elapsed_time % 60
    print(f"\nNegative Images Trained Completed in {minutes:.2f} minutes {remaining_seconds:.2f} seconds")
    print("------------------------------------\n")

    print('Total positive files after augmentation: ', Kpositive)
    print('Total negative files after augmentation: ', Knegative)
    
        

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--origenPositiveImages', type=str, help='Directory with positive (Moiré pattern) images.')
    parser.add_argument('--origenNegativeImages', type=str, help='Directory with negative (Normal) images.')

    parser.add_argument('--outputPositiveImages', type=str, help='Directory with transformed positive Images.')
    parser.add_argument('--outputNegativeImages', type=str, help='Directory with transformed negative Images.')
    
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))