import os
import sys
import argparse
import cv2
import numpy as np
from os import listdir
from os.path import join
import tensorflow as tf
from tensorflow.keras.models import load_model # type: ignore
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
    
def Scharr(img):
    image_np = img.numpy()

    scharr_x = cv2.Scharr(image_np, cv2.CV_64F, 1, 0)
    scharr_y = cv2.Scharr(image_np, cv2.CV_64F, 0, 1)

    scharr_combined = np.sqrt(scharr_x**2 + scharr_y**2)
    scharr_combined = np.uint8(scharr_combined)
    
    return scharr_combined

def Sobel(img):
    image_np = img.numpy()
    
    sobel_x = cv2.Sobel(image_np, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image_np, cv2.CV_64F, 0, 1, ksize=3)

    sobel_combined = np.sqrt(sobel_x**2 + sobel_y**2)
    sobel_combined = np.uint8(sobel_combined)
    
    return sobel_combined

def Gabor(img, ksize=31, sigma=6.0, theta=0, lambd=4.0, gamma=0.2, psi=0.0):
    image_np = img.numpy()
        
    gabor_kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_64F)
    gabor_filtered = cv2.filter2D(image_np, cv2.CV_64F, gabor_kernel)
    gabor_filtered = np.uint8(np.abs(gabor_filtered))
    
    return gabor_filtered

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

def preprocessImage(image):
    imageCrop = crop(image, HEIGHT, WIDTH)
    image = tf.image.rgb_to_grayscale(imageCrop)
    imgScharr = Scharr(image)
    imgSobel = Sobel(image)
    imgGabor = Gabor(image)
    image = tf.image.per_image_standardization(image)
    image = tf.squeeze(image, axis=-1)
    
    LL, LH, HL, HH = wavelet_transform(image)
    
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
    
    LL_resized = np.expand_dims(LL_resized, axis=0)
    LH_resized = np.expand_dims(LH_resized, axis=0)
    HL_resized = np.expand_dims(HL_resized, axis=0)
    HH_resized = np.expand_dims(HH_resized, axis=0)
    imgScharr_resized = np.expand_dims(imgScharr_resized, axis=0)
    imgSobel_resized = np.expand_dims(imgSobel_resized, axis=0)
    imgGabor_resized = np.expand_dims(imgGabor_resized, axis=0)
              
    return {
        'LL_Input': LL_resized,
        'LH_Input': LH_resized,
        'HL_Input': HL_resized,
        'HH_Input': HH_resized,
        'Scharr_Input': imgScharr_resized,
        'Sobel_Input': imgSobel_resized,
        'Gabor_Input': imgGabor_resized
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
                preprocessed_img = preprocessImage(img)
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
            'HH_Input': np.vstack([img['HH_Input'] for img in batch_images]),
            'Scharr_Input': np.vstack([img['Scharr_Input'] for img in batch_images]),
            'Sobel_Input': np.vstack([img['Sobel_Input'] for img in batch_images]),
            'Gabor_Input': np.vstack([img['Gabor_Input'] for img in batch_images]),
        }
        
        # Realizar la predicción en el batch
        predictions = model.predict_on_batch(batch_inputs)
        predicted_classes = (predictions > 0.45).astype(int)
        for img_path, prediction in zip(img_paths, predicted_classes):
            print(f'Predicción clase: {prediction[0]}\tRuta: {img_path}')
    
def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--modelPath', type=str, required=True, help='Path to the trained model.')
    parser.add_argument('--rootPath', type=str, required=True, help='Folder containing images to evaluate.')
    
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))