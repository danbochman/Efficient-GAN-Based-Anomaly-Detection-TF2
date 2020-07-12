import cv2


def display_slices(img_slices):
    """ simple function to show the result of your img_slices from function img_slice_and_labels"""
    for i, img in enumerate(img_slices):
        cv2.imshow('Image Slice ' + str(i), img)
        cv2.waitKey(0)


def bboxes_included_in_crop(vertical, horizontal, interval, bboxes):
    """
    Check whether there's a bbox inside the crop
    :param int vertical: y coordinate
    :param int horizontal: x coordinate
    :param int interval: size of height or width from coordinates
    :param list bboxes: list of bboxes in (y, x, width, height) format
    :return float: 1.0 if crop contains bbox else 0.0
    """
    for y, x, w, h in bboxes:
        cond_1 = (vertical <= y) and (y + w <= vertical + interval)
        cond_2 = (horizontal <= x) and (x + h <= horizontal + interval)
        if all([cond_1, cond_2]):
            return 1.0

    return 0.0


def img_slice_and_label(img, crop_size, bboxes=None, resize=False):
    """
    Takes an image and slices it to squares of crop_size x crop_size
    Can additionally resize the crops (usually to shrink to smaller img than crop_size)
    if bboxes from annotations are available, can label the crops 1 if bbox is present or 0 if not
    :param np.array img: array of image
    :param int crop_size: crop_size by which to slice
    :param list bboxes: list bboxes in (y, x, width, height) format
    :param bool or float resize: resize the slices after cropping (usually to shrink even more for big models)
    :return tuple (img_slices, labels): tuple of 2 lists img_slices and labels (all 0.0 if no bboxes)
    """
    width = img.shape[1]
    height = img.shape[0]
    img_slices = []
    labels = []

    v_interval = crop_size - necessary_overlap_region(width, crop_size)
    h_interval = crop_size - necessary_overlap_region(height, crop_size)

    for vertical in range(0, width - crop_size, v_interval):
        for horizontal in range(0, height - crop_size, h_interval):
            crop = img[horizontal:horizontal + crop_size, vertical:vertical + crop_size]

            if resize:
                w = int(crop.shape[1] * resize)
                h = int(crop.shape[0] * resize)
                crop = cv2.resize(crop, (width, height), interpolation=cv2.INTER_AREA)

            img_slices.append(crop)
            if bboxes:
                label = bboxes_included_in_crop(vertical, horizontal, crop_size, bboxes)
                labels.append(label)
            else:
                labels.append(0.0)

    return img_slices, labels


def necessary_overlap_region(axis, interval):
    """
    computes the amount of overlap necessary between crops to take steps to slice the image completely without missing
    edges
    :param int axis: length of axis H or W
    :param int interval: the crop_size interval
    :return int overlap_region: minimum amount of pixels necessary to overlap between to slice the image symmetrically
    without missing the edges
    """
    quotient, remainder = divmod(axis - interval, interval)
    overlap_region = int(remainder / (quotient))
    return overlap_region
