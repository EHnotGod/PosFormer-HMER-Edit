import torch
import torch.nn as nn
from einops import rearrange, repeat
from torch import Tensor
from torch.nn.modules.batchnorm import BatchNorm1d


class MaskBatchNorm2d(nn.Module):
    def __init__(self, num_features: int):
        super().__init__()
        self.bn = BatchNorm1d(num_features)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x : Tensor
            [b, d, h, w]
        mask : Tensor
            [b, 1, h, w]

        Returns
        -------
        Tensor
            [b, d, h, w]
        """
        x = rearrange(x, "b d h w -> b h w d")
        mask = mask.squeeze(1)

        not_mask = ~mask

        flat_x = x[not_mask, :]
        flat_x = self.bn(flat_x)
        x[not_mask, :] = flat_x

        x = rearrange(x, "b h w d -> b d h w")

        return x


class AttentionRefinementModule(nn.Module):
    def __init__(self, nhead: int, dc: int, cross_coverage: bool, self_coverage: bool):
        super().__init__()
        assert cross_coverage or self_coverage
        self.nhead = nhead
        self.cross_coverage = cross_coverage
        self.self_coverage = self_coverage

        if cross_coverage and self_coverage:
            in_chs = 2 * nhead
        else:
            in_chs = nhead

        self.conv = nn.Conv2d(in_chs, dc, kernel_size=5, padding=2)
        self.act = nn.ReLU(inplace=True)

        self.proj = nn.Conv2d(dc, nhead, kernel_size=1, bias=False)
        self.post_norm = MaskBatchNorm2d(nhead)

    def forward(
        self, prev_attn: Tensor, key_padding_mask: Tensor, h: int, curr_attn: Tensor,tgt_vocab: Tensor,
    ) -> Tensor:
        """
        Parameters
        ----------
        prev_attn : Tensor
            [(b * nhead), t, l]
        key_padding_mask : Tensor
            [b, l]
        h : int

        Returns
        -------
        Tensor
            [(b * nhead), t, l]
        """
        t = curr_attn.shape[1]
        mask = repeat(key_padding_mask, "b (h w) -> (b t) () h w", h=h, t=t)

        curr_attn = rearrange(curr_attn, "(b n) t l -> b n t l", n=self.nhead)
        prev_attn = rearrange(prev_attn, "(b n) t l -> b n t l", n=self.nhead)
        # b=curr_attn.shape[0] // 2
        # tgt_vocab=tgt_vocab.repeat(self.nhead,1)
        attns = []
        if self.cross_coverage:
            attns.append(prev_attn)
        if self.self_coverage:
            attns.append(curr_attn)
        attns = torch.cat(attns, dim=1)
        
        # tgt_vocab_l=tgt_vocab[:b,:]
        # mask_vocab_l=torch.logical_not(torch.logical_or(tgt_vocab_l == 110, torch.logical_or(tgt_vocab_l == 82, tgt_vocab_l == 83)))
        # tgt_vocab_r=tgt_vocab[b:,:]
        # mask_vocab_r=torch.logical_not(torch.logical_or(tgt_vocab_r == 112, torch.logical_or(tgt_vocab_r == 82, tgt_vocab_r == 83)))
        # mask_vocab=torch.cat((mask_vocab_l, mask_vocab_r), dim=0)
        # mask_vocab=mask_vocab.unsqueeze(1).repeat(1, 2*self.nhead, 1)
        tgt_vocab=tgt_vocab.unsqueeze(1).repeat(1, 2*self.nhead, 1)
        # mask_vocab = torch.logical_not(torch.logical_or(torch.logical_or(tgt_vocab == 110, tgt_vocab == 112), torch.logical_or(tgt_vocab == 82, tgt_vocab == 83)))
        # mask_vocab = torch.logical_not(torch.logical_or(tgt_vocab == 110, torch.logical_or(tgt_vocab == 82, tgt_vocab == 83)))
        mask_vocab = torch.logical_not(torch.logical_or(tgt_vocab == 7, torch.logical_or(tgt_vocab == 35, tgt_vocab == 34)))
        attns = attns*mask_vocab.unsqueeze(-1).float()
        attns = attns.cumsum(dim=2) - attns
        attns = rearrange(attns, "b n t (h w) -> (b t) n h w", h=h)

        cov = self.conv(attns)
        cov = self.act(cov)

        cov = cov.masked_fill(mask, 0.0)
        cov = self.proj(cov)

        cov = self.post_norm(cov, mask)

        cov = rearrange(cov, "(b t) n h w -> (b n) t (h w)", t=t)
        return cov


