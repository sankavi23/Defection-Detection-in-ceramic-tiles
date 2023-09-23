import cv2
import math


def convert_to_gray(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray

def gaussian_blur(gray):
    blur = cv2.GaussianBlur(gray,(5,5),0)
    return blur

def apply_thresh(blur):    
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    return thresh

def find_bound_box(thresh,image):
    x,y,w,h = cv2.boundingRect(thresh)
    cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 5)
    out = cv2.putText(image, "w={} inch,h={} inch".format(math.floor(w/96),math.floor(h/96)), (15,15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,10,10), 2)
    return out

