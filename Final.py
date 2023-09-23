

import cv2 
import numpy as np
import SizeMethods
import crackMethods





#get the image from user. save it as "originalImage" 


originalImage = cv2.imread(r'input images\input_25.jpg')

def crack_detection_plaintiles(image):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
    blur = crackMethods.gaussian_blur(image,(23,23))
    gray = crackMethods.convert_to_gray(blur)
    thresh = crackMethods.adaptive_thresh(gray)
    canny = crackMethods.canny_edge_detection(thresh)
    closing = crackMethods.morphological_closing(canny,kernel)
    dilation = crackMethods.dilation(closing,kernel)
    contours = crackMethods.find_contours(dilation)
    crackMethods.draw_contours(image,3000,contours)
    



def crack_detection_designtiles(image):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9,9))
    gray = crackMethods.convert_to_gray(image)
    filtered = crackMethods.filter(gray)
    blur = crackMethods.gaussian_blur(filtered,(31,31))  
    thresh = crackMethods.adaptive_thresh(gray)
    canny = crackMethods.canny_edge_detection(thresh)
    closing = crackMethods.morphological_closing(canny,kernel)
    dilation = crackMethods.dilation(closing,kernel)
    contours = crackMethods.find_contours(dilation)
    crackMethods.draw_contours(image,3000,contours)
    


def tile_size_calculation(image):
    gray = SizeMethods.convert_to_gray(image)
    blur = SizeMethods.gaussian_blur(gray)
    thresh = SizeMethods.apply_thresh(blur)
    out = SizeMethods.find_bound_box(thresh,image)

    return out

#show images 

crack_detection_plaintiles(originalImage)
##crack_detection_designtiles(originalImage)
cv2.imshow('Crack Detection',tile_size_calculation(originalImage))
 
