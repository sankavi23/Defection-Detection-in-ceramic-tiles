import cv2 
import numpy as np


def median_blur(image, kernel_size):
    median_blur=cv2.medianBlur(image,kernel_size)
    return median_blur

def filter(image):
    # Histogram equalization
    result_image = cv2.equalizeHist(image)

    # Bilateral filtering
    result_image = cv2.bilateralFilter(result_image, 5, 75, 75)
    return result_image


def gaussian_blur(image,kernel_size):
    gaussian_blur=cv2.GaussianBlur(image,kernel_size,0)
    return gaussian_blur

def convert_to_gray(image):
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    return gray

def adaptive_thresh(image):
    thresh = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,11,3)
    return thresh

def canny_edge_detection(image):
    canny = cv2.Canny(image,120,255,1)
    return canny

def morphological_closing(image,kernel):
    closing = cv2.morphologyEx(image,cv2.MORPH_CLOSE, kernel)
    return closing

def dilation(image,kernel):
    dilate = cv2.dilate(image,kernel,iterations=2)
    return dilate

def find_contours(image):
    contours = cv2.findContours(image,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    return contours

def draw_contours(image,min_area,contours):
    for c in contours:
        area = cv2.contourArea(c)
        if area > min_area :
            cv2.drawContours(image,[c],-1,(15,15,255),2)


    
