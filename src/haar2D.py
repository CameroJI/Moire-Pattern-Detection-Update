import pywt
import pywt.data

# computes the homography coefficients for PIL.Image.transform using point correspondences
def fwdHaarDWT2D(img):
    coeffs2 = pywt.wavedec2(img, 'bior2.2', level=3)
    LL, (HL, LH, HH) = coeffs2[0], coeffs2[1]
    return LL, LH, HL, HH