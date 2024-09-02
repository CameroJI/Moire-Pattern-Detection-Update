import os
import sys
import argparse
import cv2
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
    
    #print(model.summary())

    evaluateFolder(model, dirPath)

def preprocess_image(img):
    imageCrop = crop(img, HEIGHT, WIDTH)
    img_tensor = tf.image.rgb_to_grayscale(imageCrop)
    img_tensor = tf.image.per_image_standardization(img_tensor)
    img_tensor = tf.squeeze(img_tensor, axis=-1)
        
    LL, LH, HL, HH = wavelet_transform(img_tensor.numpy())
    
    LL_tensor = np.expand_dims(LL, axis=-1)
    LH_tensor = np.expand_dims(LH, axis=-1)
    HL_tensor = np.expand_dims(HL, axis=-1)
    HH_tensor = np.expand_dims(HH, axis=-1)
    
    LL_resized = resize(LL_tensor, HEIGHT/8, WIDTH/8)
    LH_resized = resize(LH_tensor, HEIGHT/8, WIDTH/8)
    HL_resized = resize(HL_tensor, HEIGHT/8, WIDTH/8)
    HH_resized = resize(HH_tensor, HEIGHT/8, WIDTH/8)
    
    LL_resized = np.expand_dims(LL_resized, axis=0)
    LH_resized = np.expand_dims(LH_resized, axis=0)
    HL_resized = np.expand_dims(HL_resized, axis=0)
    HH_resized = np.expand_dims(HH_resized, axis=0)
    
    return {
        'LL_Input': LL_resized,
        'LH_Input': LH_resized,
        'HL_Input': HL_resized,
        'HH_Input': HH_resized
    }

def evaluateFolder(model, image_folder, batch_size=32):
    image_files = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]
    
    if not image_files:
        print("No images found in the specified folder.")
        return
    
    positivesCount = 0
    negativesCount = 0
    num_batches = len(image_files) // batch_size + (1 if len(image_files) % batch_size != 0 else 0)

    for batch_idx in range(num_batches):
        batch_files = image_files[batch_idx * batch_size:(batch_idx + 1) * batch_size]
        batch_images = []
        img_paths = []
        
        for img_file in batch_files:
            img_path = os.path.join(image_folder, img_file)
            
            if img_path.endswith(('.jpg', '.jpeg', '.png', '.tiff')):
                img = cv2.imread(img_path)
                
                if img is None:
                    print(f"Error loading image {img_path}. Skipping...")
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                preprocessed_img = preprocess_image(img)
                batch_images.append(preprocessed_img)
                img_paths.append(img_path)
            
            # Si algunas imágenes no se cargaron correctamente
            if not batch_images:
                continue
        
        # Combinar los datos de cada batch
        batch_inputs = {
            'LL_Input': np.vstack([img['LL_Input'] for img in batch_images]),
            'LH_Input': np.vstack([img['LH_Input'] for img in batch_images]),
            'HL_Input': np.vstack([img['HL_Input'] for img in batch_images]),
            'HH_Input': np.vstack([img['HH_Input'] for img in batch_images])
        }
        
        # Realizar la predicción en el batch
        predictions = model.predict_on_batch(batch_inputs)
        predicted_classes = (predictions > 0.4).astype(int)
        for img_path, prediction in zip(img_paths, predicted_classes):
            print(f'Predicción clase: {prediction[0]}\tRuta: {img_path}')
        
        # Contar los positivos y negativos en el batch
        batch_positives = np.sum(predicted_classes == 0)
        batch_negatives = np.sum(predicted_classes == 1)
        
        # Acumulación total
        positivesCount += batch_positives
        negativesCount += batch_negatives
        
        # Imprimir conteo por batch
        print(f"Batch {batch_idx + 1}/{num_batches}: Positivos={batch_positives}, Negativos={batch_negatives}")
    
    # Imprimir conteo total
    print('Total Positivos Detectados:', positivesCount)
    print('Total Negativos Detectados:', negativesCount)
    
def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--modelPath', type=str, required=True, help='Path to the trained model.')
    parser.add_argument('--dirPath', type=str, required=True, help='Folder containing images to evaluate.')
    parser.add_argument('--height', type=int, help='Height of images to resize for model input.', default=800)
    parser.add_argument('--width', type=int, help='Width of images to resize for model input.', default=1400)
    parser.add_argument('--batch_size', type=int, help='Number of images processed by evrey iteration.', default=32)

    
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))