from enum import Enum
from typing import List

import numpy as np
from huggingface_hub import hf_hub_download
from ultralytics import YOLO


class TableDetectModel(str, Enum):
    foduucom_yolov8s = "foduucom/table-detection-and-extraction"
    keremberke_yolov8m = "keremberke/yolov8m-table-extraction"


TABLE_DETECT_MODELS = {
    TableDetectModel.foduucom_yolov8s: {
        "repo_id": "foduucom/table-detection-and-extraction",
        "filename": "best.pt",
        "table_ids": [0, 1],
    },
    TableDetectModel.keremberke_yolov8m: {
        "repo_id": "keremberke/yolov8m-table-extraction",
        "filename": "best.pt",
        "table_ids": [0, 1],
    },
}


class TableDetect:
    def __init__(self, name: TableDetectModel = TableDetectModel.keremberke_yolov8m):
        model_path = hf_hub_download(
            repo_id=TABLE_DETECT_MODELS[name]["repo_id"],
            filename=TABLE_DETECT_MODELS[name]["filename"],
        )
        self.table_ids = TABLE_DETECT_MODELS[name]["table_ids"]
        self.model = YOLO(model_path)

    def __call__(self, np_bgr_img: np.ndarray) -> List[List[int]]:
        result = self.model(np_bgr_img, conf=0.5, save_crop=False)[0]

        boxes = []
        for cls_id, box in zip(result.boxes.cls, result.boxes.xyxy):
            if cls_id in self.table_ids:
                boxes.append(box.cpu().numpy().astype(int).tolist())
        return boxes

    def vis(self, np_bgr_img: np.ndarray) -> np.ndarray:
        result = self.model(np_bgr_img, conf=0.5, save_crop=False)[0]
        vis_img = result.plot()
        return vis_img
