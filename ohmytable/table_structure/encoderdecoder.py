import torch
from torch import Tensor, nn

from .components import TokenEmbedding, PositionEmbedding


class EncoderDecoder(nn.Module):
    """Encoder decoder architecture that takes in a tabular image and generates the text output.
    Backbone serves as the image processor. There are three types of backbones: CNN, linear projection, and ConvStem.

    Args:
    ----
        backbone: tabular image processor
        encoder: transformer encoder
        decoder: transformer decoder
        vocab_size: size of the vocabulary
        d_model: feature size
        padding_idx: index of <pad> in the vocabulary
        max_seq_len: max sequence length of generated text
        dropout: dropout rate
        norm_layer: layernorm
        init_std: std in weights initialization
    """

    def __init__(
        self,
        backbone: nn.Module,
        encoder: nn.Module,
        decoder: nn.Module,
        vocab_size: int,
        d_model: int,
        padding_idx: int,
        max_seq_len: int,
        dropout: float,
        norm_layer: nn.Module,
        init_std: float = 0.02,
    ):
        super().__init__()

        self.backbone = backbone
        self.encoder = encoder
        self.decoder = decoder
        self.norm = norm_layer(d_model)
        self.token_embed = TokenEmbedding(vocab_size=vocab_size, d_model=d_model, padding_idx=padding_idx)
        self.pos_embed = PositionEmbedding(max_seq_len=max_seq_len, d_model=d_model, dropout=dropout)
        self.generator = nn.Linear(d_model, vocab_size)

    def encode(self, src: Tensor) -> Tensor:
        src_feature = self.backbone(src)
        src_feature = self.pos_embed(src_feature)
        memory = self.encoder(src_feature)
        memory = self.norm(memory)
        return memory

    def decode(
        self,
        memory: Tensor,
        tgt: Tensor,
        tgt_mask: Tensor,
        tgt_padding_mask: Tensor,
    ) -> Tensor:
        input_pos = torch.tensor([tgt.shape[1] - 1], device=tgt.device, dtype=torch.int)
        tgt = tgt[:, -1:]
        tgt_feature = self.pos_embed(self.token_embed(tgt), input_pos=input_pos)
        tgt = self.decoder(tgt_feature, memory, input_pos)

        return tgt

    def forward(
        self,
        src: Tensor,
        tgt: Tensor,
        tgt_mask: Tensor,
        tgt_padding_mask: Tensor,
    ) -> Tensor:
        memory = self.encode(src)
        tgt = self.decode(memory, tgt, tgt_mask, tgt_padding_mask)
        tgt = self.generator(tgt)

        return tgt
