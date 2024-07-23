# copyright (c) 2022 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import html
from typing import List, Dict

import numpy as np

from .ocr.ocr import OCROutput
from .table_structure.table_structure import TableStructureOutput
from .table_structure.utils import HTML_TABLE_TEMPLATE


def distance(box_1, box_2):
    x1, y1, x2, y2 = box_1
    x3, y3, x4, y4 = box_2
    dis = abs(x3 - x1) + abs(y3 - y1) + abs(x4 - x2) + abs(y4 - y2)
    dis_2 = abs(x3 - x1) + abs(y3 - y1)
    dis_3 = abs(x4 - x2) + abs(y4 - y2)
    return dis + min(dis_2, dis_3)


def compute_iou(rec1, rec2) -> float:
    """

    Args:
        rec1: (left, top, right, bottom)
        rec2: (left, top, right, bottom)

    Returns:
        float
    """
    # computing area of each rectangles
    S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
    S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])

    # computing the sum_area
    sum_area = S_rec1 + S_rec2

    # find the each edge of intersect rectangle
    left_line = max(rec1[0], rec2[0])
    right_line = min(rec1[2], rec2[2])
    top_line = max(rec1[1], rec2[1])
    bottom_line = min(rec1[3], rec2[3])

    # judge if there is an intersect
    if left_line >= right_line or top_line >= bottom_line:
        return 0.0
    else:
        intersect = (right_line - left_line) * (bottom_line - top_line)
        return (intersect / (sum_area - intersect)) * 1.0


def points_to_bbox(points: List[int]) -> List[int]:
    return [
        np.min(points[0::2]),
        np.min(points[1::2]),
        np.max(points[0::2]),
        np.max(points[1::2]),
    ]


def assign_text_box_to_table_cell(dt_boxes: List[List[int]], table_cell_boxes: List[List[int]]) -> Dict[int, List[int]]:
    # output key: table cell index
    # output value: text box indices
    matched = {}
    for i, gt_box in enumerate(dt_boxes):
        gt_box = points_to_bbox(gt_box)
        distances = []
        for j, pred_box in enumerate(table_cell_boxes):
            distances.append(
                (
                    distance(gt_box, pred_box),
                    1.0 - compute_iou(gt_box, pred_box),
                )
            )  # compute iou and l1 distance
        sorted_distances = distances.copy()
        # select det box by iou and l1 distance
        sorted_distances = sorted(sorted_distances, key=lambda item: (item[1], item[0]))
        if distances.index(sorted_distances[0]) not in matched.keys():
            matched[distances.index(sorted_distances[0])] = [i]
        else:
            matched[distances.index(sorted_distances[0])].append(i)
    return matched


def filter_ocr_result(table_structure_output: TableStructureOutput, ocr_output: OCROutput) -> OCROutput:
    # only keep the text boxes that are in the table
    new_dt_boxes = []
    new_rec_res = []
    table_xmin, table_ymin, table_xmax, table_ymax = table_structure_output.bbox()

    for box, rec in zip(ocr_output.boxes, ocr_output.contents):
        bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax = points_to_bbox(box)
        if bbox_xmin > table_xmax:
            continue
        if bbox_ymin > table_ymax:
            continue
        if bbox_xmax < table_xmin:
            continue
        if bbox_ymax < table_ymin:
            continue

        new_dt_boxes.append(box)
        new_rec_res.append(rec)

    return OCROutput(boxes=new_dt_boxes, contents=new_rec_res)


def get_full_html_output(
    table_structure_output: TableStructureOutput,
    ocr_output: OCROutput,
    filter_ocr=True,
    style_html: bool = False,
) -> str:
    if len(table_structure_output.html_tokens) == 0:
        return ""
    if filter_ocr:
        ocr_output = filter_ocr_result(table_structure_output, ocr_output)
    # key: cell box index
    # value: det boxes indices
    match = assign_text_box_to_table_cell(ocr_output.boxes, table_structure_output.boxes)

    output_html_tokens: List[str] = []
    start_token_index = 0
    for cell_box_index, end_token_index in enumerate(table_structure_output.indices):
        output_html_tokens.extend(table_structure_output.html_tokens[start_token_index : end_token_index + 1])
        # table_box_index may not exist in match,
        for det_box_index in match.get(cell_box_index, []):
            output_html_tokens.append(html.escape(ocr_output.contents[det_box_index][0]))
        start_token_index = end_token_index + 1

    if (
        table_structure_output.indices
        and table_structure_output.indices[-1] < len(table_structure_output.html_tokens) - 1
    ):
        output_html_tokens.extend(table_structure_output.html_tokens[table_structure_output.indices[-1] + 1 :])

    res = "<table>" + "".join(output_html_tokens) + "</table>"
    if style_html:
        res = HTML_TABLE_TEMPLATE.format(table=res)
    return res
