import sys
import argparse
import pywt
import numpy as np
from os import makedirs, walk
from os.path import exists
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model # type: ignore
from keras.layers import Dense # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from mCNN import createMobileModel
from modelCallbacks import BatchCheckpointCallback, EpochCheckpointCallback, CustomImageDataGenerator

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
  
    # lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    #     initial_learning_rate=initial_learning_rate,
    #     decay_steps=decay_steps,
    #     decay_rate=decay_rate,
    #     staircase=True
    # )
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate)
    
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
    
    datagen = ImageDataGenerator(
        rescale=1.0/255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.65, 1.35],
        preprocessing_function=preprocessImage
    )

    X_train = CustomImageDataGenerator(
        directory=datasetPath,
        batch_size=batch_size,
        image_size=(HEIGHT, WIDTH),
        preprocess_function=preprocessImage,
        class_mode='binary',
        classes={'Ataque': 0, 'Reales': 1}
    )
    
    class_weights = {0: 1.2, 1: 1.0}
    
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

def crop(image, target_height, target_width):
    image = tf.convert_to_tensor(image)
    
    original_height = tf.shape(image)[0]
    original_width = tf.shape(image)[1]
    
    offset_height = (original_height - target_height) // 2
    offset_width = (original_width - target_width) // 2
    
    cropped_image = image[
        offset_height:offset_height + target_height,
        offset_width:offset_width + target_width,
    ]
    
    return cropped_image

def wavelet_transform(image, wavelet='bior2.2', level=3):
    coeffs = pywt.wavedec2(image, wavelet, level=level)
    LL, (LH, HL, HH) = coeffs[0], coeffs[1]
    return LL, LH, HL, HH

def resize(component, target_height, target_width):
    component_resized = tf.image.resize(component, (target_height, target_width), method='bilinear')
    return component_resized

def preprocessImage(image):
    image = tf.image.rgb_to_grayscale(image)
    image = tf.image.per_image_standardization(image)
    image = tf.squeeze(image, axis=-1)
    
    LL, LH, HL, HH = wavelet_transform(image)
    
    wavelet_tensor = tf.stack([LL, HH, HL, HH], axis=-1)
    
    wavelet_tensor = resize(wavelet_tensor, int(HEIGHT/8), int(WIDTH/8))    
    return wavelet_tensor

def getModel(loadFlag, path):
    if loadFlag:
        model = load_model(path)
        # for layer in model.layers:
        #     layer.trainable = False

        # for layer in model.layers:
        #     if isinstance(layer, Dense):
        #         layer.trainable = True
    else:
        model = createMobileModel(height=int(HEIGHT/8), width=int(WIDTH/8), depth=1)
        
    return model

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1', 'True'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0', 'False'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

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
    
    parser.add_argument('--loadCheckPoint', type=str2bool, help='Enable Checkpoint Load', default='True')
    
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))