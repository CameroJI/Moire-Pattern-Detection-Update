import pywt
import pywt.data
import numpy as np
from math import pi
from skimage.filters import gabor

# computes the homography coefficients for PIL.Image.transform using point correspondences
def fwdHaarDWT2D(img):
    coeffs2 = pywt.dwt2(img, 'bior1.3')
    LL, (HL, LH, HH) = coeffs2
    gaborImg = gaborFilter(img)
    
    return gaborImg, LH, HL, HH

def gaborFilter(img, frequency=0.56, theta=pi/2):
    filt_real, filt_imag = gabor(np.array(img, dtype=np.float32), frequency=frequency, theta=theta)
    
    magnitude = np.sqrt(filt_real**2 + filt_imag**2)
    magnitude = (magnitude - np.min(magnitude)) / (np.max(magnitude) - np.min(magnitude)) * 255
    
    return magnitude