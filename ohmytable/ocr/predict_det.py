import numpy as np
import torch
import yaml
from huggingface_hub import hf_hub_download

from .base_ocr import BaseOCR
from .data import create_operators, transform
from .postprocess import DBPostProcess
from ..consts import HF_REPO_NAME
from ..utils import draw_det_res, Choices


class OCRDetModel(Choices):
    ppocr_v4 = "ppocr-v4"
    ppocr_v4_server = "ppocr-v4-server"


PADDLE_OCR_DET_CONFIG = {
    OCRDetModel.ppocr_v4: {
        "model": "ch_ptocr_v4_det_infer.pth",
        "config": "ch_PP-OCRv4_det_student.yml",
    },
    OCRDetModel.ppocr_v4_server: {
        "model": "ch_ptocr_v4_det_server_infer.pth",
        "config": "ch_PP-OCRv4_det_teacher.yml",
    },
}


class TextDetector(BaseOCR):
    def __init__(
        self,
        name: OCRDetModel = OCRDetModel.ppocr_v4,
        device: str = "cpu",
        limit_side_len: int = 960,
    ):
        assert name in PADDLE_OCR_DET_CONFIG
        model_path = hf_hub_download(repo_id=HF_REPO_NAME, filename=PADDLE_OCR_DET_CONFIG[name]["model"])
        config_path = hf_hub_download(repo_id=HF_REPO_NAME, filename=PADDLE_OCR_DET_CONFIG[name]["config"])
        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)

        self.device = device
        pre_process_list = [
            {
                "DetResizeForTest": {
                    "resize_long": limit_side_len,
                }
            },
            {
                "NormalizeImage": {
                    "std": [0.229, 0.224, 0.225],
                    "mean": [0.485, 0.456, 0.406],
                    "scale": "1./255.",
                    "order": "hwc",
                }
            },
            {"ToCHWImage": None},
            {"KeepKeys": {"keep_keys": ["image", "shape"]}},
        ]
        self.preprocess_op = create_operators(pre_process_list)
        self.postprocess_op = DBPostProcess(use_dilation=False, score_mode="fast", **config["PostProcess"])
        super(TextDetector, self).__init__(config["Architecture"])
        self.load_pytorch_weights(model_path)
        self.net.eval().to(self.device)

    def order_points_clockwise(self, pts):
        """
        reference from: https://github.com/jrosebr1/imutils/blob/master/imutils/perspective.py
        # sort the points based on their x-coordinates
        """
        xSorted = pts[np.argsort(pts[:, 0]), :]

        # grab the left-most and right-most points from the sorted
        # x-roodinate points
        leftMost = xSorted[:2, :]
        rightMost = xSorted[2:, :]

        # now, sort the left-most coordinates according to their
        # y-coordinates so we can grab the top-left and bottom-left
        # points, respectively
        leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
        (tl, bl) = leftMost

        rightMost = rightMost[np.argsort(rightMost[:, 1]), :]
        (tr, br) = rightMost

        rect = np.array([tl, tr, br, bl], dtype="float32")
        return rect

    def clip_det_res(self, points, img_height, img_width):
        for pno in range(points.shape[0]):
            points[pno, 0] = int(min(max(points[pno, 0], 0), img_width - 1))
            points[pno, 1] = int(min(max(points[pno, 1], 0), img_height - 1))
        return points

    def filter_tag_det_res(self, dt_boxes, image_shape):
        img_height, img_width = image_shape[0:2]
        dt_boxes_new = []
        for box in dt_boxes:
            box = self.order_points_clockwise(box)
            box = self.clip_det_res(box, img_height, img_width)
            rect_width = int(np.linalg.norm(box[0] - box[1]))
            rect_height = int(np.linalg.norm(box[0] - box[3]))
            if rect_width <= 3 or rect_height <= 3:
                continue
            dt_boxes_new.append(box)
        dt_boxes = np.array(dt_boxes_new)
        return dt_boxes

    def filter_tag_det_res_only_clip(self, dt_boxes, image_shape):
        img_height, img_width = image_shape[0:2]
        dt_boxes_new = []
        for box in dt_boxes:
            box = self.clip_det_res(box, img_height, img_width)
            dt_boxes_new.append(box)
        dt_boxes = np.array(dt_boxes_new)
        return dt_boxes

    @torch.inference_mode
    def __call__(self, img):
        origin_image_shape = img.shape
        data = {"image": img}
        data = transform(data, self.preprocess_op)
        img, shape_list = data
        if img is None:
            return None, 0
        img = np.expand_dims(img, axis=0)
        shape_list = np.expand_dims(shape_list, axis=0)
        inp = torch.from_numpy(img)
        outputs = self.net(inp.to(self.device))

        preds = {"maps": outputs["maps"].cpu().numpy()}

        post_result = self.postprocess_op(preds, shape_list)
        dt_boxes = post_result[0]["points"]
        dt_boxes = self.filter_tag_det_res(dt_boxes, origin_image_shape)
        return dt_boxes

    def vis(self, np_bgr_img: np.ndarray) -> np.ndarray:
        res = self(np_bgr_img)
        return draw_det_res(np_bgr_img, res)
