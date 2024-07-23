from typing import List, Tuple

import tokenizers as tk
from tokenizers.pre_tokenizers import PreTokenizer
from torch import Tensor

IMG_SIZE = 448
TASK_TOKENS = [
    "[table]",
    "[html]",
    "[cell]",
    "[bbox]",
    "[cell+bbox]",
    "[html+bbox]",
]
EOS_TOKEN = "<eos>"
BBOX_TOKENS = [f"bbox-{i}" for i in range(IMG_SIZE + 1)]

HTML_BBOX_HTML_TOKENS = [
    "<td></td>",
    "<td>[",
    "]</td>",
    "<td",
    ">[",
    "></td>",
    "<tr>",
    "</tr>",
    "<tbody>",
    "</tbody>",
    "<thead>",
    "</thead>",
    ' rowspan="2"',
    ' rowspan="3"',
    ' rowspan="4"',
    ' rowspan="5"',
    ' rowspan="6"',
    ' rowspan="7"',
    ' rowspan="8"',
    ' rowspan="9"',
    ' rowspan="10"',
    ' rowspan="11"',
    ' rowspan="12"',
    ' rowspan="13"',
    ' rowspan="14"',
    ' rowspan="15"',
    ' rowspan="16"',
    ' rowspan="17"',
    ' rowspan="18"',
    ' rowspan="19"',
    ' colspan="2"',
    ' colspan="3"',
    ' colspan="4"',
    ' colspan="5"',
    ' colspan="6"',
    ' colspan="7"',
    ' colspan="8"',
    ' colspan="9"',
    ' colspan="10"',
    ' colspan="11"',
    ' colspan="12"',
    ' colspan="13"',
    ' colspan="14"',
    ' colspan="15"',
    ' colspan="16"',
    ' colspan="17"',
    ' colspan="18"',
    ' colspan="19"',
    ' colspan="25"',
]

VALID_HTML_BBOX_TOKENS = [EOS_TOKEN] + HTML_BBOX_HTML_TOKENS + BBOX_TOKENS


def rescale_bbox(bbox: List[List[float]], src: Tuple[int, int], tgt: Tuple[int, int]) -> List[List[int]]:
    assert len(src) == len(tgt) == 2
    ratio = [tgt[0] / src[0], tgt[1] / src[1]] * 2
    bbox = [[int(round(i * j)) for i, j in zip(entry, ratio)] for entry in bbox]
    return bbox


def pred_token_within_range(
    pred: Tensor,
    white_list: List[int] = None,
    black_list: List[int] = None,
) -> Tensor:
    assert white_list is None or black_list is None
    if white_list:
        total = set([i for i in range(pred.shape[-1])])
        black_list = list(total.difference(set(white_list)))

    pred[..., black_list] = -float("inf")

    return pred


def html_str_to_token_list(seq: str, splitter: PreTokenizer = None) -> List[str]:
    """Convert decode output (str) to a list of tokens for constructing html table code"""

    # works for no <eos>
    seq = seq.split("<eos>")[0]

    token_black_list = ["<eos>", "<pad>", *TASK_TOKENS]
    for i in token_black_list:
        seq = seq.replace(i, "")

    if not splitter:
        splitter = tk.pre_tokenizers.Split(pattern=" ", behavior="contiguous")

    seq = splitter.pre_tokenize_str(seq)
    # only preserve the space for spanning cell tokens
    seq = [i[0] for i in seq if len(i[0].strip()) != 0 or i[1][1] - i[1][0] != 1]

    return seq


def bbox_str_to_token_list(seq: str, splitter: PreTokenizer = None) -> List[List[int]]:
    """
    Note the out could be an empty list

    return
    [[ymin, xmin, ymax, xmax],
     [ymin, xmin, ymax, xmax],
    ...
    ]
    """

    seq = seq.split("<eos>")[0]

    token_black_list = ["<eos>", "<pad>", *TASK_TOKENS]
    for i in token_black_list:
        seq = seq.replace(i, "")

    if not splitter:
        splitter = tk.pre_tokenizers.Split(pattern=" ", behavior="removed")

    seq = splitter.pre_tokenize_str(seq)
    seq = [int(i[0].split("-")[1]) for i in seq]

    rounded_seq_len = len(seq) // 4 * 4
    out = [seq[i : i + 4] for i in range(0, rounded_seq_len, 4)]
    return out


def is_valid_bbox(box: List[int]) -> bool:
    return box[2] > box[0] and box[3] > box[1]


HTML_TABLE_TEMPLATE = """<html>
    <head> <meta charset="UTF-8">
        <style>
        table, th, td {{
            border: 1px solid black;
            font-size: 10px;
            border-collapse: collapse;
        }}
        </style> 
    </head>
    <body>
       {table}
    </body> 
</html>
"""
