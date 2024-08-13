import pywt
import pywt.data
import numpy as np
from math import pi
from skimage.filters import gabor

# computes the homography coefficients for PIL.Image.transform using point correspondences
def fwdHaarDWT2D(img):
    coeffs2 = pywt.dwt2(img, 'bior1.3')
    LL, (HL, LH, HH) = coeffs2
    
    return LL, LH, HL, HH