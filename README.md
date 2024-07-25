# OhMyTable

![example](https://raw.githubusercontent.com/Sanster/OhMyTable/main/assets/example.jpg)

## Install

```bash
pip install ohmytable
```

## Quick Start

Use as a package

```python
from ohmytable import OhMyTable

image_path = "/path/to/your_image_contains_table"
ohmytable = OhMyTable(device="cpu")  # cpu/mps/cuda
htmls = ohmytable(image_path)
# The entire pipeline outputs table structure represented in HTML.
print(htmls)

# Visualize and save the results of all models in the pipeline.
from ohmytable.callback import VisualizeCallback

ohmytable(image_path, callbacks=[VisualizeCallback(image_path, "./tmp")])
```

Start a gradio web demo:

```bash
git clone https://github.com/Sanster/OhMyTable.git
cd OhMyTable
pip install gradio typer
python3 gradio_demo.py
```

## Limitation

- Table Structure Recognition model is trained with max output length 1024(about 150 table cell boxes.)
- The model effect will be better with less padding around the table image.

## Acknowledgement

- [PaddleOCR2Pytorch](https://github.com/frotms/PaddleOCR2Pytorch)
- [unitable](https://github.com/poloclub/unitable)
- [keremberke/yolov8m-table-extraction)](https://huggingface.co/keremberke/yolov8m-table-extraction)