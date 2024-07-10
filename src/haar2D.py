import numpy as np
import pywt
import pywt.data

# computes the homography coefficients for PIL.Image.transform using point correspondences
def fwdHaarDWT2D(img):
    coeffs2 = pywt.dwt2(img, 'bior1.3')
    LL, (HL, LH, HH) = coeffs2
    
    return LL, LH, HL, HH