import numpy as np
import sys
import argparse
from math import ceil
import time
from os import makedirs
from os.path import join, exists, splitext, isfile
from PIL import Image, ImageOps
import random
from mCNN import createModel_mobileNetV2
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model # type: ignore
from keras.metrics import Precision, Recall # type: ignore
from sklearn.metrics import f1_score

def custom_loss(y_true, y_pred, weight_pos=1.0, weight_neg=1.0, ssim_weight=0.1):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    # Verificar dimensiones
    print("y_true shape:", tf.shape(y_true))
    print("y_pred shape:", tf.shape(y_pred))

    # Si es necesario, redimensionar y_true para que coincida con y_pred
    if len(tf.shape(y_true)) == 4 and len(tf.shape(y_pred)) == 4:
        target_shape = tf.shape(y_pred)[1:3]  # [height, width]
        y_true_resized = tf.image.resize(y_true, target_shape)
    else:
        y_true_resized = y_true

    # Calcular la pérdida binaria ponderada
    binary_loss = tf.keras.losses.binary_crossentropy(y_true, y_pred, from_logits=False)
    false_positives = tf.reduce_sum((1 - y_true) * y_pred, axis=-1)
    true_negatives = tf.reduce_sum((1 - y_true) * (1 - y_pred), axis=-1)
    weighted_binary_loss = binary_loss + weight_pos * false_positives - weight_neg * true_negatives

    # Calcular SSIM
    ssim_value = tf.image.ssim(y_true_resized, y_pred, max_val=1.0)
    ssim_loss = 1 - ssim_value

    # Combinar la pérdida ponderada y SSIM
    combined_loss = weighted_binary_loss + ssim_weight * ssim_loss

    return combined_loss

optimizer = keras.optimizers.Adam(learning_rate=1e-3)
train_acc_metric = keras.metrics.BinaryAccuracy()
val_acc_metric = keras.metrics.BinaryAccuracy()

train_precision_metric = Precision()
train_recall_metric = Recall()
val_precision_metric = Precision()
val_recall_metric = Recall()


def main(args):
    global optimizer
    
    positiveImagePath = args.positiveImages
    negativeImagePath = args.negativeImages
    
    positiveDataImagePath = args.trainingDataPositive
    negativeDataImagePath = args.trainingDataNegative
    
    numEpochs = args.epochs
    save_epoch = args.save_epoch
    init_epoch = args.init_epoch
    save_iter = args.save_iter
    
    batch_size = args.batch_size
    
    checkpointPath = args.checkpointPath
    loadCheckPoint = args.loadCheckPoint
    
    height = args.height
    width = args.width
    
    learning_rate = args.learning_rate
    
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        
    trainIndex = createIndex(positiveDataImagePath, negativeDataImagePath)

    epochFilePath = f"{checkpointPath}/epoch.txt"
    checkpoint_path = f"{checkpointPath}/cp.keras"
    
    if not exists(checkpointPath):
        makedirs(checkpointPath)
    
    if loadCheckPoint:
        model = load_model(checkpoint_path)
        
    else:
        model = createModel_mobileNetV2(height=height, width=width, depth=7)

    model.compile(
        loss=lambda y_true, y_pred: custom_loss(y_true, y_pred, weight_pos=21.5, weight_neg=1.0, ssim_weight=0.1),
        optimizer='adam',
        metrics=['accuracy', 'precision', 'recall', 'f1_score']
    )
    
    epoch = epochFileValidation(epochFilePath, loadCheckPoint, init_epoch)
    
    model = trainModel(trainIndex, positiveImagePath, negativeImagePath, positiveDataImagePath, negativeDataImagePath, epoch, numEpochs, 
        epochFilePath, save_epoch, save_iter, batch_size, height, width, checkpoint_path, model)

def readAndScaleImage(f, customStr, imgOrig, trainImagePath, X_train, X_index, Y, sampleIndex, sampleVal, height, width):
    f = str(f)
    fileName = splitext(f)[0]
    fLL = f"{splitext(f.replace(fileName, fileName + customStr + '_LL'))[0]}.png"
    fLH = f"{splitext(f.replace(fileName, fileName + customStr + '_LH'))[0]}.png"
    fHL = f"{splitext(f.replace(fileName, fileName + customStr + '_HL'))[0]}.png"
    fHH = f"{splitext(f.replace(fileName, fileName + customStr + '_HH'))[0]}.png"
    
    try:
        imgLL = PreprocessImage(join(trainImagePath, fLL), width, height).convert('L')
        imgLH = PreprocessImage(join(trainImagePath, fLH), width, height).convert('L')
        imgHL = PreprocessImage(join(trainImagePath, fHL), width, height).convert('L')
        imgHH = PreprocessImage(join(trainImagePath, fHH), width, height).convert('L')
                
        # DATA AUGMENTATION FOR TRAINING
        imgLL = channelAugmentation(imgLL)
        imgLH = channelAugmentation(imgLH)
        imgHL = channelAugmentation(imgHL)
        imgHH = channelAugmentation(imgHH)
        
    except Exception as e:
        print(f"Error: Couldn\'t read the file {fileName}. Make sure only images are present in the folder")
        print('Exception:', e)
        return None

    imgLL = np.array(imgLL)
    imgLH = np.array(imgLH)
    imgHL = np.array(imgHL)
    imgHH = np.array(imgHH)
    
    imgOrig = imgAugmentation(imgOrig)
    imgOrig_np = np.array(imgOrig)

    X_train[sampleIndex, :, :, :3] = imgOrig_np[:, :, :3]

    X_train[sampleIndex, :, :, 3] = imgLL
    X_train[sampleIndex, :, :, 4] = imgLH
    X_train[sampleIndex, :, :, 5] = imgHL
    X_train[sampleIndex, :, :, 6] = imgHH 
    
    Y[sampleIndex, 0] = sampleVal
    X_index[sampleIndex, 0] = sampleIndex

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

    return img

def imgAugmentation(image):
    image_np = np.array(image)

    if len(image_np.shape) != 3 or image_np.shape[2] != 3:
        raise ValueError("The input image must be a 3-D image with 3 channels (RGB).")
    
    image_tf = tf.convert_to_tensor(image_np, dtype=tf.float32)
    
    # Aplicar aumentos
    image_tf = tf.image.random_brightness(image_tf, max_delta=0.15)
    image_tf = tf.image.random_contrast(image_tf, lower=0.85, upper=1.15)
    
    image_np = image_tf.numpy().astype(np.uint8)
    image_pil = Image.fromarray(image_np)

    # Rotación aleatoria entre -20 y 20 grados
    angle = random.uniform(-20, 20)
    image_pil = image_pil.rotate(angle, resample=Image.BICUBIC, expand=True)

    # Obtener dimensiones originales
    original_width, original_height = image.size
    # Recortar la imagen rotada al tamaño original
    left = (image_pil.width - original_width) / 2
    top = (image_pil.height - original_height) / 2
    right = (image_pil.width + original_width) / 2
    bottom = (image_pil.height + original_height) / 2

    image_pil = image_pil.crop((left, top, right, bottom))
    
    image_np = np.array(image_pil)
    image_tf = tf.convert_to_tensor(image_np, dtype=tf.float32)
    
    # Clip valores entre 0 y 255, y convertir a uint8
    image_tf = tf.clip_by_value(image_tf, 0, 255)
    image_tf = tf.cast(image_tf, dtype=tf.uint8)

    # Convertir de vuelta a PIL Image
    image_augmented = Image.fromarray(image_tf.numpy())
    
    return image_augmented

def channelAugmentation(img):
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

def createIndex(posPath, negPath):
    posList = readLstFile(posPath, 'positiveFiles.lst')
    negList = readLstFile(negPath, 'negativeFiles.lst')

    datasetList = [(i, 1) for i in posList]
    datasetList.extend((i, 0) for i in negList)

    random.shuffle(datasetList)
        
    return datasetList

def readLstFile(path, filename):
    listPath = join(path, filename)
    with open(listPath, 'r') as file:
        lines = [line.strip() for line in file]

    return lines

def epochFileValidation(path, loadFlag, init_epoch):
    if not exists(path) or not loadFlag:
        with open(path, 'w') as epochFile:
            if init_epoch == 0:
                epoch = 1
                epochFile.write("1")
            else:
                epoch = init_epoch
                epochFile.write(str(epoch))
    elif init_epoch == 0:
        with open(path, 'r') as epochFile:
            epoch = int(epochFile.read())
    else:
        epoch = init_epoch

    return epoch

def saveEpochFile(epochFilePath, epoch):
    with open(epochFilePath, 'w') as epochFile:
        epochFile.write(str(epoch))
        print(f"\nEpoch Save: {epoch}")
        
@tf.function
def train_step(model, X_train, Y_train):
    with tf.GradientTape() as tape:
        logits = model(X_train, training=True)
        loss_value = tf.reduce_mean(custom_loss(Y_train, logits))

    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    
    train_acc_metric.update_state(Y_train, logits)
    train_precision_metric.update_state(Y_train, logits)
    train_recall_metric.update_state(Y_train, logits)
    
    return loss_value
        
def f1Score(precision, recall):
    return 2 * ((precision * recall) / (precision + recall + tf.keras.backend.epsilon()))

def trainModel(listInput, posJpgPath, negJpgPath, posPath, negPath, epoch, epochs, epochFilePath, save_epoch, save_iter, batch_size, height, width, checkpoint_path, model):
    epoch -= 1
    n = len(listInput)
    start_time_full = time.time()
    for i in range(epochs - epoch):
        saveEpochFile(epochFilePath, i + 1)
        print(f"epoch: {i + epoch + 1}/{epochs}\n")
        start_time = time.time()
        for j in range(ceil(n/batch_size)):
            start, end = defineEpochRange(j, batch_size, n)            
            X_train, Y_train = getBatch(listInput, posJpgPath, negJpgPath, posPath, negPath, start, end, batch_size, height, width)
            
            loss = train_step(model, X_train, Y_train)
            
            print("------------------------------------")
            print(f"Training {end - start} images ({j + 1}/{ceil(n/batch_size)})", end='\t')
            print(f'start: {start}\tend: {end}\tTotal Images:{len(listInput)}\tLoss: {loss*100:.2f}%')
            train_acc = train_acc_metric.result()
            train_precision = train_precision_metric.result()
            train_recall = train_recall_metric.result()
            f1 = f1Score(train_precision, train_recall)
            
            print(f'\nTraining acc over batch: {float(train_acc.numpy())*100:.2f}%')
            print(f'Training precision over batch: {float(train_precision.numpy())*100:.2f}%')
            print(f'Training recall over batch: {float(train_recall.numpy())*100:.2f}%')
            print(f'F1-score over batch: {float(f1.numpy())*100:.2f}%')

            # Reset metrics at the end of each batch
            train_acc_metric.reset_state()
            train_precision_metric.reset_state()
            train_recall_metric.reset_state()

            # Validación
            val_logits = model(X_train, training=False)
            val_acc_metric.update_state(Y_train, val_logits)
            val_precision_metric.update_state(Y_train, val_logits)
            val_recall_metric.update_state(Y_train, val_logits)
            val_acc = val_acc_metric.result()
            val_precision = val_precision_metric.result()
            val_recall = val_recall_metric.result()
            val_f1 = f1Score(val_precision, val_recall)
            
            print(f'Validation acc: {float(val_acc.numpy())*100:.2f}%')
            print(f'Validation precision: {float(val_precision.numpy())*100:.2f}%')
            print(f'Validation recall: {float(val_recall.numpy())*100:.2f}%')
            print(f'Validation F1-score: {float(val_f1.numpy())*100:.2f}%')

            # Reset validation metrics at the end of each batch
            val_acc_metric.reset_state()
            val_precision_metric.reset_state()
            val_recall_metric.reset_state()
            
            if save_iter != 0 and (j + 1) % save_iter == 0:
                saveModel(model, checkpoint_path)
            
        end_time = time.time()
        elapsed_time = end_time - start_time
        minutes = elapsed_time // 60
        remaining_seconds = elapsed_time % 60
        print(f"Batch time: {int(minutes)} minutes {remaining_seconds:.2f} seconds")

        if (i + 1) % save_epoch == 0:
            saveModel(model, checkpoint_path)
        print("------------------------------------")
    saveModel(model, checkpoint_path)

    end_time_full = time.time()
    elapsed_time = end_time_full - start_time_full
    hours = elapsed_time // 3600
    elapsed_time %= 3600
    minutes, seconds = divmod(elapsed_time, 60)
    print(f"\nTotal training time: {int(hours)} hours, {int(minutes)} minutes, {seconds:.2f} seconds.")

    return model
        
def saveModel(model, checkpoint_path):
    print("Saving model... ", end='')
    model.save(checkpoint_path, include_optimizer=False)
    print("Model Saved.")
            

def createElements(batch_size, height, width, multiply):
    totalBatchSize = batch_size*multiply
    X_train = np.zeros((totalBatchSize, height, width, 7))
    X_index = np.zeros((totalBatchSize, 1))
    Y = np.zeros((totalBatchSize, 1))
    
    #return X_LL.astype(np.float32), X_LH.astype(np.float32), X_HL.astype(np.float32), X_HH.astype(np.float32), Y
    return X_train, X_index, Y

def defineEpochRange(epoch, batch_size, n):
    start = 0 if epoch*batch_size >= n else epoch*batch_size
    end = min(start + batch_size, n)
    
    return start, end

def getBatch(listInput, posJpgPath, negJpgPath, posPath, negPath, start, end, batch_size, height, width):
    X_train, X_index, Y = createElements(batch_size, height, width, 3)

    sampleIndex = 0
    for f in listInput[start:end]:
        file = str(f[0])
        y = int(f[1])
        pathTiff = posPath if y == 1 else negPath
        pathJpg = posJpgPath if y == 1 else negJpgPath
        
        img = PreprocessImage(join(pathJpg, file), width, height)
        ret = readAndScaleImage(file, '', img, pathTiff, X_train, X_index, Y, sampleIndex, y, height, width)
        if ret == True:
            sampleIndex += 1

        #read 180deg rotated data
        ret = readAndScaleImage(file, '_45', PreprocessImage(img.rotate(45, expand=True), width, height), pathTiff, X_train, X_index, Y, sampleIndex, y, height, width)
        if ret == True:
            sampleIndex += 1

        #read 180deg FLIP data
        ret = readAndScaleImage(file, '_90', PreprocessImage(img.rotate(90, expand=True), width, height), pathTiff, X_train, X_index, Y, sampleIndex, y, height, width)
        if ret == True:
            sampleIndex += 1
    
    return X_train, Y

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1', 'True'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0', 'False'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--positiveImages', type=str, help='Directory with original positive (Moiré pattern) images.')
    parser.add_argument('--negativeImages', type=str, help='Directory with original negative (Normal) images.')
        
    parser.add_argument('--trainingDataPositive', type=str, help='Directory with transformed positive (Moiré pattern) images.')
    parser.add_argument('--trainingDataNegative', type=str, help='Directory with transformed negative (Normal) images.')
    
    parser.add_argument('--checkpointPath', type=str, help='Directory for model Checkpoint', default='./checkpoint/')
    
    parser.add_argument('--epochs', type=int, help='Number of epochs for training', default=10)
    parser.add_argument('--save_epoch', type=int, help='Number of epochs to save the model', default=10)
    parser.add_argument('--init_epoch', type=int, help='Initial epoch for model training (if 0, starts with the saved epoch file)', default=0)
    parser.add_argument('--save_iter', type=int, help='Number of iterations to save the model', default=0)

    parser.add_argument('--batch_size', type=int, help='Batch size for epoch in training', default=32)
    parser.add_argument('--height', type=int, help='Image height resize', default=800)
    parser.add_argument('--width', type=int, help='Image width resize', default=1400)
    
    parser.add_argument('--learning_rate', type=float, help='Model learning rate for iteration', default=1e-3)
    
    parser.add_argument('--loadCheckPoint', type=str2bool, help='Enable Checkpoint Load', default='True')
    
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))