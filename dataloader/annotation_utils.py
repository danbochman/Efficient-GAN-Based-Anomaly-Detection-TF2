import json

import numpy as np


def load_annotation_file(ann_path):
    """
    Loads a supervise.ly annotation file.

    :param str ann_path: Full path to the annotation file.
    :return dict: The parsed content of the annotation.
    """
    with open(ann_path) as f:
        return json.load(f)


def annotation_to_bboxes_ltwh(ann, classes='defect'):
    """
    Returns a (y, x, width, height) bbox for each object in the annotation.

    :param dict ann: An image annotation given in supervise.ly format
    :param tuple[str]|list[str]|None classes: The object classes to be extracted (e.g. 'defect' or 'roi').
                                              None means any.
    :rtype List[(int, int, int, int)]
    """

    def obj_to_bbox(obj):
        p = np.array(obj['points']['exterior'])
        mn = np.min(p, axis=0)
        mx = np.max(p, axis=0)
        return tuple(mn) + tuple(mx - mn)

    def filter_class(obj):
        return obj['classTitle'] in classes if classes is not None else True

    return list(map(obj_to_bbox, filter(filter_class, ann['objects'])))
