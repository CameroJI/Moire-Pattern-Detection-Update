import sys
import argparse
import pywt
import numpy as np
import cv2
from os import makedirs, walk
from os.path import exists
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from mCNN import createModelElements
from modelCallbacks import BatchCheckpointCallback, EpochCheckpointCallback, CustomImageDataGenerator
from utils import crop, waveletFunction, resize, Scharr, Sobel, Gabor


HEIGHT = 800
WIDTH = 1400

def main(args):
    global optimizer, WIDTH, HEIGHT
    
    datasetPath = args.datasetPath
    
    numEpochs = args.epochs
    save_iter = args.save_iter
    
    batch_size = args.batch_size
    
    checkpointPath = args.checkpointPath
    loadCheckPoint = args.loadCheckPoint
    
    HEIGHT = args.height
    WIDTH = args.width
    image_size = (HEIGHT, WIDTH)
        
    initial_learning_rate = args.learning_rate
    final_learning_rate = 1e-5
    decay_steps = countImg(datasetPath) // batch_size
    decay_rate = (final_learning_rate / initial_learning_rate) ** (1 / decay_steps)

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=initial_learning_rate,
        decay_steps=decay_steps,
        decay_rate=decay_rate,
        staircase=True
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    
    if not exists(checkpointPath):
        makedirs(checkpointPath)
            
    checkpointPathModel = f"{checkpointPath}/cp.keras"
    checkpointPathBatch = f"{checkpointPath}/cp_checkpoint.keras"
    
    model = getModel(loadCheckPoint, checkpointPathModel)

    model.compile(
        loss='binary_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy', 'recall'])
    
    batchCheckpointCallback = BatchCheckpointCallback(batchesNumber=save_iter, path=checkpointPathBatch)
    epochCheckpointCallback = EpochCheckpointCallback(path=checkpointPathModel)

    X_train = CustomImageDataGenerator(
        directory=datasetPath,
        batch_size=batch_size,
        image_size=(HEIGHT, WIDTH),
        preprocess_function=preprocessImage,
        class_mode='binary',
        classes={'Reales': 0, 'Ataque': 1}
    )
    
    class_weights = {0: 1.0, 1: 1.0}
    
    model.fit(
        X_train, 
        epochs=numEpochs,
        callbacks=[epochCheckpointCallback, batchCheckpointCallback], 
        class_weight=class_weights
        )

def countImg(directory):
    image_extensions = ('.jpg', '.jpeg', '.png')
    total_images = 0
    
    for root, dirs, files in walk(directory):
        for file in files:
            if file.lower().endswith(image_extensions):
                total_images += 1
    
    return total_images

def preprocessImage(image):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, max_delta=0.3)
    image = tf.image.random_contrast(image, lower=0.65, upper=1.35)
    
    imageCrop = crop(image, HEIGHT, WIDTH)
    image = tf.image.rgb_to_grayscale(imageCrop)
    imgScharr = Scharr(image)
    imgSobel = Sobel(image)
    imgGabor = Gabor(image)
    image = tf.image.per_image_standardization(image)
    image = tf.squeeze(image, axis=-1)
    
    LL, LH, HL, HH = waveletFunction(image)
    
    LL_tensor = np.expand_dims(LL, axis=-1)
    LH_tensor = np.expand_dims(LH, axis=-1)
    HL_tensor = np.expand_dims(HL, axis=-1)
    HH_tensor = np.expand_dims(HH, axis=-1)
    imgScharr_tensor = np.expand_dims(imgScharr, axis=-1)
    imgSobel_tensor = np.expand_dims(imgSobel, axis=-1)
    imgGabor_tensor = np.expand_dims(imgGabor, axis=-1)
    
    LL_resized = resize(LL_tensor, HEIGHT/8, WIDTH/8)
    LH_resized = resize(LH_tensor, HEIGHT/8, WIDTH/8)
    HL_resized = resize(HL_tensor, HEIGHT/8, WIDTH/8)
    HH_resized = resize(HH_tensor, HEIGHT/8, WIDTH/8)
    imgScharr_resized = resize(imgScharr_tensor, HEIGHT/8, WIDTH/8)
    imgSobel_resized= resize(imgSobel_tensor, HEIGHT/8, WIDTH/8)
    imgGabor_resized = resize(imgGabor_tensor, HEIGHT/8, WIDTH/8)
        
    return {
        'LL_Input': LL_resized,
        'LH_Input': LH_resized,
        'HL_Input': HL_resized,
        'HH_Input': HH_resized,
        'Scharr_Input': imgScharr_resized,
        'Sobel_Input': imgSobel_resized,
        'Gabor_Input': imgGabor_resized
    }

def getModel(loadFlag, path):
    if loadFlag:
        model = load_model(path)
    else:
        model = createModelElements(height=int(HEIGHT/8), width=int(WIDTH/8), depth=1)
        
    return model

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--datasetPath', type=str, help='Directory with dataset images folders.')
    
    parser.add_argument('--checkpointPath', type=str, help='Directory for model Checkpoint', default='./checkpoint/')
    
    parser.add_argument('--epochs', type=int, help='Number of epochs for training', default=10)
    parser.add_argument('--save_iter', type=int, help='Number of iterations to save the model', default=0)

    parser.add_argument('--batch_size', type=int, help='Batch size for epoch in training', default=32)
    parser.add_argument('--height', type=int, help='Image height resize', default=800)
    parser.add_argument('--width', type=int, help='Image width resize', default=1400)
    
    parser.add_argument('--learning_rate', type=float, help='Model learning rate for iteration', default=1e-3)
    
    parser.add_argument('--loadCheckPoint', action='store_true', default=False, help='load Checkpoint Model')

    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))