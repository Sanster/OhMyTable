import base64
from io import BytesIO

import gradio as gr
import numpy as np
from typer import Typer
from PIL import Image

from ohmytable import OhMyTable
from ohmytable.callback import Callback
from ohmytable.ocr import OCROutput
from ohmytable.ocr.predict_det import OCRDetModel
from ohmytable.ocr.predict_rec import OCRRecModel
from ohmytable.table_structure.table_structure import TableStructureOutput
from ohmytable.utils import draw_det_res

app = Typer(pretty_exceptions_show_locals=False)


class GradioVisCallback(Callback):
    def __init__(self):
        self.vis_imgs = []

    def on_table_structure_output(
        self,
        table_index: int,
        table_img: np.ndarray,
        ocr_output: OCROutput,
        table_structure_output: TableStructureOutput,
    ):
        vis_img = draw_det_res(table_img, table_structure_output.boxes, color=(255, 0, 0))
        vis_img = draw_det_res(vis_img, ocr_output.boxes, color=(0, 200, 0))
        self.vis_imgs.append(vis_img)


def img_tag(img: np.ndarray):
    img = Image.fromarray(img)
    buffered = BytesIO()
    img = img.convert("RGB")
    img.save(buffered, format="JPEG")
    img_b64_str = base64.b64encode(buffered.getvalue()).decode()
    return f'<img src="data:image/jpeg;base64,{img_b64_str}" alt="user upload image" style="max-height: 400px;"/>'


@app.command()
def main(device: str = "cpu", port: int = 8000):
    ohmytable = OhMyTable(
        det_model=OCRDetModel.ppocr_v4_server,
        rec_model=OCRRecModel.ppocr_v4_ch_server,
        device=device,
    )

    def predict(img, detect_table):
        gradio_vis_callback = GradioVisCallback()
        htmls = ohmytable(img, style_html=True, detect_table=detect_table, callbacks=[gradio_vis_callback])
        res = ""
        for i, (vis_img, html) in enumerate((zip(gradio_vis_callback.vis_imgs, htmls))):
            vis_img = vis_img[:, :, ::-1]
            res += f"<h1>Table {i + 1}</h1>"
            res += img_tag(vis_img)
            res += "<h2>OhMyTable Recognition Result</h2>"
            res += html
        return res

    with gr.Blocks() as demo:
        with gr.Row():
            gr.Markdown("""
            # OhMyTable Demo
            GitHub: https://github.com/Sanster/ohmytable
            - Blue box: table cell result
            - Green box: ocr text detection result
            """)
        with gr.Row():
            with gr.Column():
                detect_table = gr.Checkbox(True, label="Detect Table")
                run_button = gr.Button("Run")
                origin_img = gr.Image(type="pil")
                gr.Examples(
                    examples=[
                        ["./assets/demo2.png", True],
                        ["./assets/demo1.jpg", False],
                        ["./tests/table_example.png", False],
                    ],
                    inputs=[origin_img, detect_table],
                )

            with gr.Column():
                with gr.Group():
                    html_result = gr.HTML("")

            run_button.click(predict, [origin_img, detect_table], html_result)

        demo.launch(server_name="0.0.0.0", server_port=port, share=False)


if __name__ == "__main__":
    app()
