import zipfile
from typing import List, Tuple
import time
import copy

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import torch.optim as optim
from torch import FloatTensor, LongTensor

from Pos_Former.datamodule import Batch, vocab, label_make_muti
from Pos_Former.model.posformer import PosFormer
from Pos_Former.utils.utils import ExpRateRecorder, Hypothesis, ce_loss_all, to_bi_tgt_out

class LitPosFormer(pl.LightningModule):
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
        beam_size: int,
        max_len: int,
        alpha: float,
        early_stopping: bool,
        temperature: float,
        learning_rate: float,
        patience: int,
        distill_alpha: float = 0.1,
    ):
        super().__init__()
        self.save_hyperparameters()

        # 主模型
        self.model = PosFormer(
            d_model=d_model,
            growth_rate=growth_rate,
            num_layers=num_layers,
            nhead=nhead,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            dc=dc,
            cross_coverage=cross_coverage,
            self_coverage=self_coverage,
        )
        # 冻结教师模型
        self.teacher_decoder = copy.deepcopy(self.model.decoder)
        self.teacher_posdecoder = copy.deepcopy(self.model.posdecoder)
        for p in list(self.teacher_decoder.parameters()) + list(self.teacher_posdecoder.parameters()):
            p.requires_grad = False

        self.exprate_recorder = ExpRateRecorder()

    def forward(
        self, img: FloatTensor, img_mask: LongTensor, tgt: LongTensor, logger
    ) -> Tuple[FloatTensor, FloatTensor, FloatTensor]:
        # 不改接口，返回 (seq_logits, layer_logits, pos_logits)
        return self.model(img, img_mask, tgt, logger)

    def training_step(self, batch: Batch, _):
        tgt, out = to_bi_tgt_out(batch.indices, self.device)
        # 模型前向
        out_hat, out_hat_layer, out_hat_pos = self(batch.imgs, batch.mask, tgt, self.trainer.logger)

        # 准备老师输出
        with torch.no_grad():
            feat, m = self.model.encoder(batch.imgs, batch.mask)
            feat = torch.cat((feat, feat), 0)
            m = torch.cat((m, m), 0)
            out_t, _ = self.teacher_decoder(feat, m, tgt)
            tgt_list = tgt.cpu().tolist()
            muti_labels = label_make_muti.tgt2muti_label(tgt_list)
            muti_tensor = torch.FloatTensor(muti_labels).to(self.device)
            out_layer_t, out_pos_t, _ = self.teacher_posdecoder(feat, m, tgt, muti_tensor)

        # 真实标签与伪标签
        layer_num, final_pos = label_make_muti.out2layernum_and_pos(tgt.cpu().tolist())
        ln_t = torch.LongTensor(layer_num).to(self.device)
        fp_t = torch.LongTensor(final_pos).to(self.device)

        # 主监督：CE + 多标签监督
        loss_ce, loss_ln, loss_fp = ce_loss_all(
            out_hat, out,
            out_hat_layer, ln_t,
            out_hat_pos, fp_t
        )

        # 蒸馏损失
        T = max(1.0, self.hparams.temperature)
        kl = F.kl_div(
            F.log_softmax(out_hat / T, dim=-1),
            F.softmax(out_t / T, dim=-1),
            reduction='batchmean'
        ) * (T * T)
        layer_distill = F.mse_loss(out_hat_layer, out_layer_t)
        pos_distill = F.mse_loss(out_hat_pos, out_pos_t)

        α = self.hparams.distill_alpha
        total_loss = (
            loss_ce + 0.25 * loss_ln + 0.25 * loss_fp
            + α * (kl + layer_distill + pos_distill)
        ) / (1.5 + 3 * α)

        # 保持原有日志
        self.log("train_loss", loss_ce, on_epoch=True, sync_dist=True, prog_bar=True)
        self.log("train_loss_layernum", loss_ln, on_epoch=True, sync_dist=True, prog_bar=True)
        self.log("train_loss_pos", loss_fp, on_epoch=True, sync_dist=True, prog_bar=True)
        # 新增蒸馏日志
        self.log("train_distill_loss", kl + layer_distill + pos_distill, on_epoch=True, sync_dist=True)

        return total_loss

    def validation_step(self, batch: Batch, _):
        tgt, out = to_bi_tgt_out(batch.indices, self.device)
        out_hat, out_hat_layer, out_hat_pos = self(batch.imgs, batch.mask, tgt, self.trainer.logger)

        layer_num, final_pos = label_make_muti.out2layernum_and_pos(tgt.cpu().tolist())
        ln_t = torch.LongTensor(layer_num).to(self.device)
        fp_t = torch.LongTensor(final_pos).to(self.device)

        loss_ce, loss_ln, loss_fp = ce_loss_all(
            out_hat, out,
            out_hat_layer, ln_t,
            out_hat_pos, fp_t
        )

        self.log("val_loss", loss_ce, on_epoch=True, sync_dist=True, prog_bar=True)
        self.log("val_loss_layernum", loss_ln, on_epoch=True, sync_dist=True, prog_bar=True)
        self.log("val_loss_pos", loss_fp, on_epoch=True, sync_dist=True, prog_bar=True)

        hyps = self.approximate_joint_search(batch.imgs, batch.mask)
        self.exprate_recorder([h.seq for h in hyps], batch.indices)
        self.log("val_ExpRate", self.exprate_recorder, on_epoch=True, prog_bar=True)

    def test_step(self, batch: Batch, _):
        start = time.time()
        hyps = self.approximate_joint_search(batch.imgs, batch.mask)
        t = time.time() - start
        self.exprate_recorder([h.seq for h in hyps], batch.indices)
        self.log('batch_inference_time', t)
        return batch.img_bases, [vocab.indices2label(h.seq) for h in hyps], t

    def test_epoch_end(self, test_outputs):
        total = sum(x[2] for x in test_outputs)
        print(f"Total Inference Time: {total:.2f}s")
        expr = self.exprate_recorder.compute()
        print(f"Validation ExpRate: {expr:.4f}")
        with zipfile.ZipFile("result.zip", "w") as z:
            for bases, preds, _ in test_outputs:
                for b, p in zip(bases, preds):
                    z.writestr(f"{b}.txt", f"%{b}\n${p}$")

    def approximate_joint_search(self, img: FloatTensor, mask: LongTensor) -> List[Hypothesis]:
        return self.model.beam_search(img, mask, **self.hparams)

    def configure_optimizers(self):
        optimizer = optim.SGD(
            self.parameters(),
            lr=self.hparams.learning_rate,
            momentum=0.9,
            weight_decay=1e-4,
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.25,
            patience=max(1, self.hparams.patience // self.trainer.check_val_every_n_epoch),
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_ExpRate", "interval": "epoch", "frequency": self.trainer.check_val_every_n_epoch, "strict": True}}