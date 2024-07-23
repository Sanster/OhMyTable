import sys

import cv2
from pathlib import Path

from ohmytable.callback import VisualizeCallback
from ohmytable.ocr import OCR
from ohmytable.ocr.predict_rec import TextRecognizer
from ohmytable.ocr.predict_det import TextDetector
from ohmytable import OhMyTable
from ohmytable.table_structure import TableStructure
from ohmytable.utils import draw_det_res

CURRENT_DIR = Path(__file__).parent
rec_ch_example = str(CURRENT_DIR / "rec_ch_example.png")
rec_en_example = str(CURRENT_DIR / "rec_en_example.png")
table_example = str(CURRENT_DIR / "table_example.png")
full_page_example = str(CURRENT_DIR / "full_page.png")


def is_mac():
    return sys.platform == "darwin"


device = "mps" if is_mac() else "cpu"


def test_rec():
    res_model = TextRecognizer(
        name="ppocr-v4-ch",
        limited_max_width=1024,
        device=device,
    )
    img = cv2.imread(rec_ch_example)
    res = res_model([img])
    print(res)
    assert res[0][0] == "其他运营生产设备及建筑物"


def test_en_rec():
    res_model = TextRecognizer(
        name="ppocr-v4-en",
        limited_max_width=1024,
        device=device,
    )
    img = cv2.imread(rec_en_example)
    res = res_model([img])
    print(res)
    assert res[0][0] == "Let me show you the library in our school."


def test_det():
    det_model = TextDetector(
        name="ppocr-v4",
        device=device,
    )
    img = cv2.imread(table_example)
    res = det_model(img)
    vis_img = draw_det_res(img, res)
    cv2.imwrite("det_result.jpg", vis_img)
    assert len(res) == 47


def test_ocr():
    ocr_model = OCR(device=device)
    img = cv2.imread(table_example)
    ocr_output = ocr_model(img)
    print(ocr_output.boxes)
    print(ocr_output.contents)


def test_table_structure():
    table_structure = TableStructure(device)
    table_structure_output = table_structure.generate(table_example)

    assert len(table_structure_output.boxes) == 47

    img = cv2.imread(table_example)
    vis_img = table_structure_output.plot(img)

    for i, html_token in enumerate(table_structure_output.html_tokens):
        try:
            box_index = table_structure_output.indices.index(i)
            print(table_structure_output.boxes[box_index])
        except:
            pass

    # cv2.imwrite("table_structure_result.jpg", vis_img)


def test_table_and_ocr():
    ohmytable = OhMyTable(device=device)
    table_image_paths = list((CURRENT_DIR / "images").glob("*.png"))
    for img_p in table_image_paths:
        ohmytable(img_p, detect_table=False, callbacks=[VisualizeCallback(img_p, CURRENT_DIR / "tmp")])


def test_full_page():
    ohmytable = OhMyTable(device=device)
    full_htmls = ohmytable(full_page_example, callbacks=[VisualizeCallback(full_page_example, CURRENT_DIR / "tmp")])
    print(full_htmls)
