from functools import partial
from typing import List, Union
from dataclasses import dataclass

import cv2
import torch
from huggingface_hub import hf_hub_download
from loguru import logger
from tokenizers import Tokenizer
from torch import nn
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from PIL.Image import Image as PILImage

from .components import ImgLinearBackbone, Encoder
from .gpt_fast_decoder import GPTFastDecoder
from .encoderdecoder import EncoderDecoder
from .utils import (
    pred_token_within_range,
    EOS_TOKEN,
    VALID_HTML_BBOX_TOKENS,
    IMG_SIZE,
    BBOX_TOKENS,
    html_str_to_token_list,
    is_valid_bbox,
    rescale_bbox,
)
from ..consts import HF_REPO_NAME


@dataclass
class TableStructureOutput:
    html_tokens: List[str]
    boxes: List[List[int]]  # x1, y1, x2, y2
    indices: List[int]  # index of bbox to html_tokens

    def plot(self, img: np.ndarray):
        vis_img = img.copy()
        for box in self.boxes:
            cv2.rectangle(vis_img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        return vis_img

    def bbox(self) -> List[int]:
        xmin = 99999
        ymin = 99999
        xmax = 0
        ymax = 0
        for box in self.boxes:
            xmin = min(xmin, box[0])
            ymin = min(ymin, box[1])
            xmax = max(xmax, box[2])
            ymax = max(ymax, box[3])
        return [xmin, ymin, xmax, ymax]


transform = transforms.Compose(
    [
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.86597056, 0.88463002, 0.87491087],
            std=[0.20686628, 0.18201602, 0.18485524],
        ),
    ]
)


class TableStructure:
    def __init__(self, device):
        model_path = hf_hub_download(repo_id=HF_REPO_NAME, filename="table_structure_240722_1024.pth")
        vocab_path = hf_hub_download(repo_id=HF_REPO_NAME, filename="vocab_html_bbox.json")

        self.vocab = Tokenizer.from_file(vocab_path)
        self.prefix_token_id = self.vocab.token_to_id("[html+bbox]")
        self.eos_id = self.vocab.token_to_id(EOS_TOKEN)
        self.device = device
        self.max_seq_len = 1024
        self.img_size = IMG_SIZE
        self.token_white_list = [self.vocab.token_to_id(i) for i in VALID_HTML_BBOX_TOKENS]
        self.bbox_token_ids = set([self.vocab.token_to_id(i) for i in BBOX_TOKENS])
        self.bbox_close_html_token = self.vocab.token_to_id("]</td>")

        d_model = 768
        patch_size = 16
        nhead = 12
        activation = "gelu"
        norm_first = True
        n_encoder_layer = 12
        n_decoder_layer = 4

        backbone = ImgLinearBackbone(d_model=d_model, patch_size=patch_size)
        encoder = Encoder(
            d_model=d_model,
            nhead=nhead,
            dropout=0,
            activation=activation,
            norm_first=norm_first,
            nlayer=n_encoder_layer,
            ff_ratio=4,
        )
        decoder = GPTFastDecoder(
            d_model=d_model,
            nhead=nhead,
            dropout=0,
            activation=activation,
            norm_first=norm_first,
            nlayer=n_decoder_layer,
            ff_ratio=4,
        )

        self.model = EncoderDecoder(
            backbone=backbone,
            encoder=encoder,
            decoder=decoder,
            vocab_size=self.vocab.get_vocab_size(),
            d_model=d_model,
            padding_idx=self.vocab.token_to_id("<pad>"),
            max_seq_len=self.max_seq_len,
            dropout=0,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
        )
        state_dict = torch.load(model_path, map_location="cpu")
        self.model.load_state_dict(state_dict)
        self.model.eval().to(device)

    @torch.inference_mode()
    def generate(
        self,
        image: Union[str, np.ndarray, PILImage],
    ) -> TableStructureOutput:
        # np.ndarray should be RGB
        if not isinstance(image, np.ndarray):
            image = Image.open(image).convert("RGB")
        else:
            image = Image.fromarray(image)
        origin_height = image.height
        origin_width = image.width

        with torch.device(self.device):
            image = transform(image).unsqueeze(0).to(self.device)
            self.model.decoder.setup_caches(
                max_batch_size=image.shape[0],
                max_seq_length=self.max_seq_len,
                dtype=image.dtype,
            )
            context = torch.tensor([self.prefix_token_id], dtype=torch.int32).repeat(1, 1)
        memory = self.model.encode(image)

        box_token_count = 0
        for _ in range(self.max_seq_len):
            eos_flag = [self.eos_id in k for k in context]
            if all(eos_flag):
                break

            logits = self.model.decode(memory, context, tgt_mask=None, tgt_padding_mask=None)
            logits = self.model.generator(logits)[:, -1, :]

            logits = pred_token_within_range(
                logits.detach(),
                white_list=self.token_white_list,
            )

            probs = F.softmax(logits, dim=-1)
            _, next_tokens = probs.topk(1)
            # TODO: support batch
            if next_tokens[0] in self.bbox_token_ids:
                box_token_count += 1
                if box_token_count > 4:
                    # force stop output bbox tokens
                    next_tokens = torch.tensor([self.bbox_close_html_token], dtype=torch.int32)
                    box_token_count = 0

            context = torch.cat([context, next_tokens], dim=1)

        pred_html = context[0]
        pred_html = pred_html.detach().cpu().numpy()
        pred_html = self.vocab.decode(pred_html, skip_special_tokens=False)
        pred_html = html_str_to_token_list(pred_html)

        html_tokens: List[str] = []
        bboxes: List[List[int]] = []
        indices: List[int] = []

        box = []
        for token in pred_html:
            if token.startswith("bbox"):
                box.append(int(token.split("-")[-1]))
                if len(box) == 4:
                    if is_valid_bbox(box):
                        bboxes.append(box)
                        indices.append(len(html_tokens) - 1)
                    else:
                        logger.warning(f"invalid bbox: {box}")
                    box = []
                continue
            token = token.replace("[", "")
            token = token.replace("]", "")
            html_tokens.append(token)
        bboxes = rescale_bbox(
            bboxes,
            src=(self.img_size, self.img_size),
            tgt=(origin_width, origin_height),
        )

        # make sure html token end with </tr> or </td> or <td></td>
        valid_end_html_tokens = ["</tr>", "</td>", "<td></td>"]
        if html_tokens[-1] not in valid_end_html_tokens:
            # find last </tr> / </td> token in html_tokens
            last_valid_end_html_token = -1
            for i in range(len(html_tokens) - 1, -1, -1):
                if html_tokens[i] in valid_end_html_tokens:
                    last_valid_end_html_token = i
                    break
            if last_valid_end_html_token != -1:
                if not indices or (indices and last_valid_end_html_token >= indices[-1]):
                    html_tokens = html_tokens[: last_valid_end_html_token + 1]

        assert len(bboxes) == len(indices), f"{len(bboxes)} != {len(indices)}"

        return TableStructureOutput(html_tokens=html_tokens, boxes=bboxes, indices=indices)
