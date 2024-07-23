from typing import List, Union, Optional

import numpy as np
from pathlib import Path
from PIL import Image
from PIL.Image import Image as PILImage

from .callback import Callback
from .matcher import get_full_html_output
from .ocr.ocr import OCR
from .ocr.predict_det import OCRDetModel
from .ocr.predict_rec import OCRRecModel
from .table_detect import TableDetect
from .table_detect.table_detect import TableDetectModel
from .table_structure import TableStructure


class OhMyTable:
    def __init__(
        self,
        table_detect_model: Optional[TableDetectModel] = TableDetectModel.keremberke_yolov8m,
        det_model: OCRDetModel = OCRDetModel.ppocr_v4,
        rec_model: OCRRecModel = OCRRecModel.ppocr_v4_ch,
        device: str = "cpu",
    ):
        self.table_detect = TableDetect(name=table_detect_model) if table_detect_model else None
        self.ocr_model = OCR(det_model=det_model, rec_model=rec_model, device=device)
        self.table_structure = TableStructure(device=device)

    def __call__(
        self,
        image: Union[np.ndarray, str, Path, PILImage],
        detect_table: bool = True,
        style_html: bool = False,
        callbacks: List[Callback] = [],
    ) -> List[str]:
        if isinstance(image, str) or isinstance(image, Path):
            image = Image.open(image).convert("RGB")
            np_bgr_img = np.array(image)[:, :, ::-1]
        elif isinstance(image, PILImage):
            np_bgr_img = np.array(image)[:, :, ::-1]
        else:
            np_bgr_img = image

        if self.table_detect and detect_table:
            table_boxes = self.table_detect(np_bgr_img)

            for it in callbacks:
                it.on_table_detect(np_bgr_img, table_boxes)
        else:
            table_boxes = [[0, 0, np_bgr_img.shape[1], np_bgr_img.shape[0]]]

        full_html_output = []
        for table_index, table_box in enumerate(table_boxes):
            left, top, right, bottom = table_box
            table_img = np_bgr_img[top:bottom, left:right]
            ocr_output = self.ocr_model(table_img)

            for it in callbacks:
                it.on_table_ocr_output(table_index, table_img, ocr_output)

            table_structure_output = self.table_structure.generate(table_img[:, :, ::-1])

            for it in callbacks:
                it.on_table_structure_output(table_index, table_img, ocr_output, table_structure_output)

            res = get_full_html_output(table_structure_output, ocr_output, style_html=style_html)

            full_html_output.append(res)

        return full_html_output
