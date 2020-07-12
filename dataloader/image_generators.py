import glob
import random

import cv2
import numpy as np

from dataloader.annotation_utils import load_annotation_file, annotation_to_bboxes_ltwh
from dataloader.cogwheel_slicer import img_slice_and_label
from dataloader.preprocessing import stack_and_expand, center_and_scale


def crop_generator(img_ann_list, batch_size=64, crop_size=256, shuffle=True, normalize=True, resize=False, repeat=True):
    """
    Given a list of tuples (img_path, ann_path) yield batches of img_batch, label_batches.
    Can shuffle the data, normalize the images, resize and be one-shot or infinite.
    :param list img_ann_list: list of tuples (str img_path, str ann_path)
    :param int batch_size: max batch size to yield (max may not be reached if bigger than img slices by crops
    :param int crop_size: in what crop_size x crops_size shapes to slice the image (assuming it's bigger)
    :param bool shuffle: whether to shuffle the data
    :param bool normalize: whether to transform [0, 255] images to [-1, 1]
    :param bool or float resize: resize the slices after cropping (usually to shrink even more for big models)
    :param bool repeat: one shot iterator or infinite generator
    :yield tuple: (img_batch, label_batch)
    """
    repeats = 2 ** 32 if repeat else 1
    random.seed(42)

    for i in range(repeats):

        if shuffle:
            random.shuffle(list(img_ann_list))

        for img_path, ann_path in img_ann_list:
            img = cv2.imread(img_path, 0)

            if normalize:
                img = center_and_scale(img)

            ann = load_annotation_file(ann_path)
            bboxes = annotation_to_bboxes_ltwh(ann)
            img_slices, labels = img_slice_and_label(img, crop_size, bboxes, resize=resize)
            img_slices = stack_and_expand(img_slices)
            labels = np.array(labels)
            for i in range(0, img_slices.shape[0], batch_size):
                img_batch = img_slices[i:i + batch_size]
                label_batch = labels[i:i + batch_size]
                yield img_batch, label_batch


def separate_images_by_label(img_list, ann_list):
    """
    filter globs of img list and ann list by whether they have bboxes in their annotation.json file
    :param list img_list, ann_list: lists of string datapaths from glob.glob
    :return tuple: normal_img_list (no bboxes), defect_img_list (bboxes of defects)
    """
    normal_img_list = []
    defect_img_list = []
    for img_path, ann_path, in zip(img_list, ann_list):
        ann_parsed = load_annotation_file(ann_path)
        bboxes = annotation_to_bboxes_ltwh(ann_parsed)
        if len(bboxes) == 0:
            normal_img_list.append((img_path, ann_path))
        elif len(bboxes) > 0:
            defect_img_list.append((img_path, ann_path))

    return normal_img_list, defect_img_list


def train_val_test_image_generator(data_path, batch_size=128, crop_size=128, ext="png", normalize=True, resize=False,
                                   val_frac=0.0):
    # load img and annotation filepath recursively from folder
    img_list = [img for img in sorted(glob.glob(data_path + "**/*." + ext, recursive=True))]
    ann_list = [img for img in sorted(glob.glob(data_path + "**/*." + "json", recursive=True))]

    # separate images/annotation by label
    normal_img_ann_list, defect_img_ann_list = separate_images_by_label(img_list, ann_list)

    # split to train/val/test
    test_generator = crop_generator(defect_img_ann_list, batch_size=batch_size, crop_size=crop_size,
                                    normalize=normalize,
                                    resize=resize,
                                    repeat=False,
                                    shuffle=False)
    if val_frac:
        # split to train/val (we only need normal imgs)
        num_images = len(normal_img_ann_list)
        train_val_split = int(num_images * (1 - val_frac))
        train_img_ann_list = normal_img_ann_list[:train_val_split]
        val_img_ann_list = normal_img_ann_list[train_val_split:]

        train_generator = crop_generator(train_img_ann_list, batch_size=batch_size, crop_size=crop_size,
                                         normalize=normalize,
                                         repeat=True,
                                         shuffle=True,
                                         resize=resize)
        val_generator = crop_generator(val_img_ann_list, batch_size=batch_size, crop_size=crop_size,
                                       normalize=normalize,
                                       repeat=True,
                                       shuffle=False,
                                       resize=resize)

        return train_generator, val_generator, test_generator

    else:
        # if there's not need for validation (e.g. for GANs) return only train and test generators
        train_generator = crop_generator(normal_img_ann_list, batch_size=batch_size, crop_size=crop_size,
                                         normalize=normalize,
                                         repeat=True,
                                         shuffle=True,
                                         resize=resize)

        return train_generator, test_generator


if __name__ == '__main__':
    data_path = "path/to/dataset"
    train_generator, val_generator, test_generator = train_val_test_image_generator(data_path, val_frac=0.2)
    for i in range(10):
        train_img_sample = train_generator.__next__()
        val_img_sample = val_generator.__next__()
        test_img_sample = test_generator.__next__()
        print(train_img_sample[0].shape)
        print(val_img_sample[0].shape)
        print(test_img_sample[0].shape)
