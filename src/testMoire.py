import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore
from tensorflow.keras.preprocessing.image import img_to_array # type: ignore
import matplotlib.pyplot as plt
from trainMoire import crop, wavelet_transform, resize

HEIGHT = 800
WIDTH = 1400

def main(args):
    global HEIGHT, WIDTH
    
    modelPath = args.modelPath
    dirPath = args.dirPath
    
    HEIGHT = args.height
    WIDTH = args.width

    model = load_model(modelPath)

    evaluateFolder(model, dirPath)

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(HEIGHT, WIDTH))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)[0]
    
    img_tensor = tf.image.rgb_to_grayscale(img_tensor)
    img_tensor = tf.image.per_image_standardization(img_tensor)
    img_tensor = tf.squeeze(img_tensor, axis=-1)
    
    img_tensor = crop(img_tensor, HEIGHT, WIDTH)
    
    LL, LH, HL, HH = wavelet_transform(img_tensor.numpy())
    
    LL_tensor = np.expand_dims(LL, axis=-1)
    LH_tensor = np.expand_dims(LH, axis=-1)
    HL_tensor = np.expand_dims(HL, axis=-1)
    HH_tensor = np.expand_dims(HH, axis=-1)
    
    LL_resized = resize(LL_tensor, HEIGHT/8, WIDTH/8)
    LH_resized = resize(LH_tensor, HEIGHT/8, WIDTH/8)
    HL_resized = resize(HL_tensor, HEIGHT/8, WIDTH/8)
    HH_resized = resize(HH_tensor, HEIGHT/8, WIDTH/8)
    
    return {
        'LL_Input': LL_resized,
        'LH_Input': LH_resized,
        'HL_Input': HL_resized,
        'HH_Input': HH_resized
    }

def evaluateFolder(model, image_folder):
    image_files = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]
    
    for img_file in image_files:
        img_path = os.path.join(image_folder, img_file)
        preprocessed_img = preprocess_image(img_path)
        
        # Realizar la predicción
        prediction = model.predict(preprocessed_img)
        predicted_class = np.argmax(prediction, axis=-1)[0]
        
        plt.imshow(image.load_img(img_path))
        plt.title(f"Predicted: {predicted_class}")
        plt.axis('off')
        plt.show()
        
        # Esperar a que el usuario presione una tecla para continuar
        input("Press Enter to continue to the next image...")
    
def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--modelPath', type=str, required=True, help='Path to the trained model.', default='./')
    parser.add_argument('--dirPath', type=str, required=True, help='Folder containing images to evaluate.', default='./')
    parser.add_argument('--height', type=int, required=True, help='Height of images to resize for model input.', default=800)
    parser.add_argument('--width', type=int, required=True, help='Width of images to resize for model input.', width=1400)
    
    return parser.parse_args(argv)