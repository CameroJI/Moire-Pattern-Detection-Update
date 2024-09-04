import cv2
import numpy as np
import pywt
import tensorflow as tf

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

def waveletFunction(img):
    coeffs2 = pywt.wavedec2(img, 'bior2.2', level=3)
    LL, (HL, LH, HH) = coeffs2[0], coeffs2[1]
    return LL, LH, HL, HH

def resize(component, target_height, target_width):
    component_resized = tf.image.resize(component, (int(target_height), int(target_width)), method='bilinear')
    return component_resized