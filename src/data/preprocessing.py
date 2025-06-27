from skimage.transform import resize
import numpy as np

def normalize(x):
    return x / 255.0

def resize_image(img, size=(32, 32)):
    return resize(img, size, anti_aliasing=True)