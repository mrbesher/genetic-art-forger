import cv2 as cv
import numpy as np
from os import path

def read_binary_img(filename = 'assets/ex1.png'):
    img = cv.imread(filename, cv.IMREAD_GRAYSCALE)
    _, im_bw = cv.threshold(img, 128, 255, cv.THRESH_BINARY)
    return np.invert(im_bw.astype(dtype=bool))

def plot_image(image, window_name = 'image') -> None:
    img = np.invert(image)
    img = img.astype(np.uint8)  #convert to an unsigned byte
    img *= 255
    cv.imshow(window_name,img)
    cv.waitKey(1)

def save_image(image, filename, foldername = 'generations', text = '', color = (150, 150, 150)) -> None:
    img = np.invert(image)
    img = img.astype(np.uint8)  #convert to an unsigned byte
    img *= 255
    img = cv.resize(img, (800,800), interpolation = cv.INTER_AREA)
    img = cv.putText(img, text, (10, img.shape[0] // 20), cv.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv.imwrite(path.join(foldername, filename), img)


def leave(arr, filename:str = 'example.png') -> None:
    """
    gets an array of bool values to save into image
    """
    output_img = np.invert(arr)
    output_img = output_img.astype(np.uint8)  #convert to an unsigned byte
    output_img *= 255
    cv.imwrite(filename, output_img)
    cv.destroyAllWindows()