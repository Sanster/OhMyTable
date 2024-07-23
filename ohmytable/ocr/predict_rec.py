import cv2
import numpy as np
import math
import torch
import yaml
from huggingface_hub import hf_hub_download

from .base_ocr import BaseOCR
from .postprocess import CTCLabelDecode
from ..consts import HF_REPO_NAME
from ..utils import Choices


class OCRRecModel(Choices):
    ppocr_v4_ch = "ppocr-v4-ch"
    ppocr_v4_ch_server = "ppocr-v4-ch-server"
    ppocr_v4_en = "ppocr-v4-en"


PADDLE_OCR_REC_CONFIG = {
    OCRRecModel.ppocr_v4_ch: {
        "model": "ch_ptocr_v4_rec_infer.pth",
        "config": "ch_PP-OCRv4_rec.yml",
        "charset": "ppocr_keys_v1.txt",
        "height": 48,
    },
    OCRRecModel.ppocr_v4_ch_server: {
        "model": "ch_ptocr_v4_rec_server_infer.pth",
        "config": "ch_PP-OCRv4_rec_hgnet.yml",
        "charset": "ppocr_keys_v1.txt",
        "height": 48,
    },
    OCRRecModel.ppocr_v4_en: {
        "model": "en_ptocr_v4_rec_infer.pth",
        "config": "en_PP-OCRv4_rec.yml",
        "charset": "en_dict.txt",
        "height": 48,
    },
}


class TextRecognizer(BaseOCR):
    def __init__(
        self,
        name: OCRRecModel = OCRRecModel.ppocr_v4_ch,
        limited_max_width: int = 1024,
        device: str = "cpu",
    ):
        assert (
            name in PADDLE_OCR_REC_CONFIG
        ), f"Unsupported model name: {name}, available: {PADDLE_OCR_REC_CONFIG.keys()}"
        model_path = hf_hub_download(repo_id=HF_REPO_NAME, filename=PADDLE_OCR_REC_CONFIG[name]["model"])
        config_path = hf_hub_download(repo_id=HF_REPO_NAME, filename=PADDLE_OCR_REC_CONFIG[name]["config"])
        charset_path = hf_hub_download(
            repo_id=HF_REPO_NAME,
            filename=PADDLE_OCR_REC_CONFIG[name]["charset"],
        )
        self.height = PADDLE_OCR_REC_CONFIG[name]["height"]
        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)
        self.device = device
        self.limited_min_width = 16
        self.limited_max_width = limited_max_width
        self.postprocess_op = CTCLabelDecode(character_dict_path=charset_path, use_space_char=True)
        weights = self.read_pytorch_weights(model_path)
        self.out_channels = self.get_out_channels(weights)
        char_num = len(getattr(self.postprocess_op, "character"))
        config["Architecture"]["Head"]["out_channels_list"] = {
            "CTCLabelDecode": char_num,
        }
        super(TextRecognizer, self).__init__(config["Architecture"], out_channels=self.out_channels)

        self.load_state_dict(weights)
        self.net.eval().to(self.device)

    def resize_norm_img(self, img, max_wh_ratio):
        imgC = 3
        imgH = self.height
        imgW = 320
        assert imgC == img.shape[2]
        max_wh_ratio = max(max_wh_ratio, imgW / imgH)
        imgW = int((imgH * max_wh_ratio))
        imgW = max(min(imgW, self.limited_max_width), self.limited_min_width)
        h, w = img.shape[:2]
        ratio = w / float(h)
        ratio_imgH = math.ceil(imgH * ratio)
        ratio_imgH = max(ratio_imgH, self.limited_min_width)
        if ratio_imgH > imgW:
            resized_w = imgW
        else:
            resized_w = int(ratio_imgH)
        resized_image = cv2.resize(img, (resized_w, imgH))
        resized_image = resized_image.astype("float32")
        resized_image = resized_image.transpose((2, 0, 1)) / 255
        resized_image -= 0.5
        resized_image /= 0.5
        padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
        padding_im[:, :, 0:resized_w] = resized_image
        return padding_im

    @torch.inference_mode
    def __call__(self, img_list, batch_size: int = 4):
        img_num = len(img_list)
        # Calculate the aspect ratio of all text bars
        width_list = []
        for img in img_list:
            width_list.append(img.shape[1] / float(img.shape[0]))
        # Sorting can speed up the recognition process
        indices = np.argsort(np.array(width_list))

        # rec_res = []
        rec_res = [["", 0.0]] * img_num
        for beg_img_no in range(0, img_num, batch_size):
            end_img_no = min(img_num, beg_img_no + batch_size)
            norm_img_batch = []
            max_wh_ratio = 0
            for ino in range(beg_img_no, end_img_no):
                # h, w = img_list[ino].shape[0:2]
                h, w = img_list[indices[ino]].shape[0:2]
                wh_ratio = w * 1.0 / h
                max_wh_ratio = max(max_wh_ratio, wh_ratio)
            for ino in range(beg_img_no, end_img_no):
                norm_img = self.resize_norm_img(img_list[indices[ino]], max_wh_ratio)
                norm_img = norm_img[np.newaxis, :]
                norm_img_batch.append(norm_img)
            norm_img_batch = np.concatenate(norm_img_batch)
            norm_img_batch = norm_img_batch.copy()

            inp = torch.from_numpy(norm_img_batch)
            inp = inp.to(self.device)
            prob_out = self.net(inp)

            if isinstance(prob_out, list):
                preds = [v.cpu().numpy() for v in prob_out]
            else:
                preds = prob_out.cpu().numpy()

            rec_result = self.postprocess_op(preds)
            for rno in range(len(rec_result)):
                rec_res[indices[beg_img_no + rno]] = rec_result[rno]
        return rec_res
