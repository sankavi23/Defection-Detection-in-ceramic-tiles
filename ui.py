import PySimpleGUI as sg
import cv2 
import numpy as np
import SizeMethods
import crackMethods


def select_image():
    image_path = '';
    while True:
        image_path = sg.popup_get_file('Select image')
        if image_path == '':
            sg.popup('Please select an image')

        if image_path != '':
            break
    image = cv2.imread(image_path)
    return image

def mark_crack(image):
    image_bw = image
    output_image = np.ndarray.copy(image)
    layout_sliders = [
        [
            sg.Text('Kernel 1'),
            sg.Slider(range=(1, 100),
                      default_value=23,
                      size=(50,15),
                      orientation='horizontal', key='-KERNEL_1-', enable_events=True)
        ],
        [
            sg.Text('Kernel 2'),
            sg.Slider(range=(1, 100),
                      default_value=7,
                      size=(50, 15),
                      orientation='horizontal', key='-KERNEL_2-', enable_events=True)
        ],
        [
            sg.In(key='-CONTOUR_AREA-', default_text='100'),
            sg.Button(button_text='Set Contour Area', key='-SET_CONTOUR-')
        ],
        [
            sg.Text('Show crack on input image'), sg.Button('Show')
        ]
    ]

    layout_im = [
        [
            sg.Image(data='', key='-IMAGE-'),

        ]
    ]

    window_im = sg.Window(title='Output Image', layout=layout_im, finalize=True)

    im_new = cv2.imencode('.png', image)[1].tobytes()
    window_im['-IMAGE-'].update(data=im_new)
    window_sliders = sg.Window(title='Editing Parameters', layout=layout_sliders)
    contour_area = 100
    while True:

        window_im.read(timeout=10)
        event_sliders, values_sliders = window_sliders.read()

        if event_sliders == sg.WIN_CLOSED:
            break
        elif event_sliders == '-SET_CONTOUR-':
            area = values_sliders['-CONTOUR_AREA-']
            contour_area = validate_contour_area(area)
        
        kernel_1 = values_sliders['-KERNEL_1-']
        KERNEL1=(int(kernel_1),int(kernel_1))
        kernel_2= values_sliders['-KERNEL_2-']
        KERNEL2=(int(kernel_2),int(kernel_2))
           
        crack_detection(image=image_bw,kernel_1=KERNEL1, kernel_2=KERNEL2)
        output_image = tile_size_calculation(image_bw)
        output_image_png = cv2.imencode('.png', output_image)[1].tobytes()
        window_im['-IMAGE-'].update(data=output_image_png)


def crack_detection(image,kernel_1,kernel_2):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,kernel_2)
    blur = crackMethods.gaussian_blur(image,kernel_1)
    gray = crackMethods.convert_to_gray(blur)
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


def validate_contour_area(input_string):
    try:
        float(input_string)
    except ValueError:
        sg.popup_error('Invalid input\nPlease Try Again\n\n\nContour Area set to 100', title='Error')
        return 100
    else:
        return float(input_string)

im = select_image()
mark_crack(im)
