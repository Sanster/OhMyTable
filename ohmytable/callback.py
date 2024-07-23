from typing import List, Union

import cv2
import numpy as np
from pathlib import Path

from .matcher import get_full_html_output
from .ocr import OCROutput
from .table_structure.table_structure import TableStructureOutput
from .utils import draw_det_res


class Callback:
    def on_table_detect(self, img: np.ndarray, table_boxes: List[List[int]]):
        pass

    def on_table_ocr_output(self, table_index: int, table_img: np.ndarray, ocr_output: OCROutput):
        pass

    def on_table_structure_output(
        self,
        table_index: int,
        table_img: np.ndarray,
        ocr_output: OCROutput,
        table_structure_output: TableStructureOutput,
    ):
        pass


class VisualizeCallback(Callback):
    def __init__(self, img_p: Union[Path, str], save_dir: Union[Path, str]):
        self.img_p = Path(img_p)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)

    def on_table_detect(self, img: np.ndarray, table_boxes: List[List[int]]):
        table_vis_img = draw_det_res(img, table_boxes)
        cv2.imwrite(str(self.save_dir / f"{self.img_p.stem}_table_detect.jpg"), table_vis_img)

    def on_table_ocr_output(self, table_index: int, table_img: np.ndarray, ocr_output: OCROutput):
        table_vis_img = draw_det_res(table_img, ocr_output.boxes)
        cv2.imwrite(
            str(self.save_dir / f"{self.img_p.stem}_table_ocr_det_{table_index}.jpg"),
            table_vis_img,
        )

    def on_table_structure_output(
        self,
        table_index: int,
        table_img: np.ndarray,
        ocr_output: OCROutput,
        table_structure_output: TableStructureOutput,
    ):
        res = get_full_html_output(table_structure_output, ocr_output, style_html=True)
        with open(self.save_dir / f"{self.img_p.stem}_table_{table_index}.html", "w", encoding="utf-8") as f:
            f.write(res)

        cell_boxes_vis_img = draw_det_res(table_img, table_structure_output.boxes)
        cv2.imwrite(str(self.save_dir / f"{self.img_p.stem}_table_cell_det_{table_index}.jpg"), cell_boxes_vis_img)
