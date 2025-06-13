from typing import List ,Tuple
import numpy as np
import pytorch_lightning as pl
import torch
from torch import FloatTensor, LongTensor
import os
from Pos_Former.utils.utils import Hypothesis

from .decoder import Decoder , PosDecoder
from .encoder import Encoder
from Pos_Former.datamodule import vocab , label_make_muti

class PosFormer(pl.LightningModule):
    def __init__(
        self,
        d_model: int,
        growth_rate: int,
        num_layers: int,
        nhead: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        dropout: float,
        dc: int,
        cross_coverage: bool,
        self_coverage: bool,
    ):
        super().__init__()
        self.encoder = Encoder(
            d_model=d_model, growth_rate=growth_rate, num_layers=num_layers
        )

        self.middle_proj = torch.nn.Sequential(
            torch.nn.LayerNorm(d_model),
            torch.nn.Linear(d_model, d_model),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout)
        )

        self.decoder = Decoder(
            d_model=d_model,
            nhead=nhead,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            dc=dc,
            cross_coverage=cross_coverage,
            self_coverage=self_coverage,
        )

        self.posdecoder = PosDecoder(
            d_model=d_model,
            nhead=nhead,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            dc=dc,
            cross_coverage=cross_coverage,
            self_coverage=self_coverage
        )

        self.fusion_norm = torch.nn.LayerNorm(d_model)

        self.save_path = 'attn_PosFormer'

    def forward(
        self, img: FloatTensor, img_mask: LongTensor, tgt: LongTensor, logger
    ) -> Tuple[FloatTensor,FloatTensor,FloatTensor]:
        feature, mask = self.encoder(img, img_mask)  # [b, t, d]

        feature = self.middle_proj(feature)  # Add projection enhancement

        feature = torch.cat((feature, feature), dim=0)  # [2b, t, d]
        mask = torch.cat((mask, mask), dim=0)

        feature = self.fusion_norm(feature)

        tgt_list = tgt.cpu().numpy().tolist()
        muti_labels = label_make_muti.tgt2muti_label(tgt_list)
        muti_labels_tensor = torch.FloatTensor(muti_labels).cuda()  # [2b,l,5]

        out, _ = self.decoder(feature, mask, tgt)
        out_layernum , out_pos, _ = self.posdecoder(feature, mask, tgt, muti_labels_tensor)

        return out, out_layernum, out_pos  # [2b,l,vocab_size], [2b,l,5], [2b,l,6]

    def beam_search(
        self,
        img: FloatTensor,
        img_mask: LongTensor,
        beam_size: int,
        max_len: int,
        alpha: float,
        early_stopping: bool,
        temperature: float,
        **kwargs,
    ) -> List[Hypothesis]:
        feature, mask = self.encoder(img, img_mask)  # [b, t, d]

        feature = self.middle_proj(feature)  # Match enhancement during inference
        feature = self.fusion_norm(feature)

        seq_out = self.decoder.beam_search(
            [feature], [mask], beam_size, max_len, alpha, early_stopping, temperature
        )

        return seq_out
