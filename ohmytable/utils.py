from enum import Enum

import cv2
import numpy as np

class Choices(str, Enum):
    @classmethod
    def values(cls):
        return [member.value for member in cls]


def draw_det_res(img, boxes, color=(200, 100, 0), thickness=1):
    vis_img = img.copy()
    for box in boxes:
        box = np.asarray(box).astype(int)
        if len(box) == 8:
            box = box.reshape(-1, 2)

        if box.shape == (4, 2):
            cv2.polylines(vis_img, [box], True, color=color, thickness=thickness)
        elif len(box) == 4:
            cv2.rectangle(
                vis_img,
                (int(box[0]), int(box[1])),
                (int(box[2]), int(box[3])),
                color=color,
                thickness=thickness,
            )
    return vis_img
