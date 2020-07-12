import numpy as np


def center_and_scale(img):
    """
    Transforms an image in uint8 format [0, 255] to float32 [-1, 1]
    """
    img = img / 255
    img = 2 * img - 1
    return img


def stack_and_expand(img_lst):
    """
    prepares a list of grayscale images to a numpy array with a redundant color channel for Keras models
    """
    img_slices = np.stack(img_lst, axis=0)
    img_slices = np.expand_dims(img_slices, axis=-1).astype(np.float)
    return img_slices
