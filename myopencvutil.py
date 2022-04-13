import cv2 as cv
import numpy as np

def read_binary_img(filename = 'assets/ex1.png'):
    img = cv.imread(filename, cv.IMREAD_GRAYSCALE)
    _, im_bw = cv.threshold(img, 128, 255, cv.THRESH_BINARY)
    return np.invert(im_bw.astype(dtype=bool))

def plot_image(image) -> None:
    img = np.invert(image)
    img = img.astype(np.uint8)  #convert to an unsigned byte
    img *= 255
    cv.imshow('image',img)
    cv.waitKey(1)

def leave(arr, filename:str = 'example.png') -> None:
    """
    gets an array of bool values to save into image
    """
    output_img = np.invert(arr)
    output_img = output_img.astype(np.uint8)  #convert to an unsigned byte
    output_img *= 255
    cv.imwrite(filename, output_img)
    cv.destroyAllWindows()