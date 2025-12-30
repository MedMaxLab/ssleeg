import math
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "TransformeegEncoder"
    "TransformEEG"
]


def _reset_seed(seed):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)


class PositionalEncoding(nn.Module):

    def __init__(
        self,
        d_model: int,
        max_len: int = 256,
        n: int = 10000
    ):
        super(PositionalEncoding, self).__init__()
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term1 = torch.pow(n, torch.arange(0, math.ceil(d_model/2))/d_model)
        if d_model%2 == 0:
            div_term2 = div_term1
        else:
            div_term2 = div_term1[:-1]

        print(div_term1.shape, div_term2.shape)
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position/div_term1)
        pe[0, :, 1::2] = torch.cos(position/div_term2)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[0,:x.size(1)]


# ==========================
#       TransformEEG
# ==========================
class EEGTokenizer(nn.Module):

    def __init__(
        self,
        Chans,
        D1 = 2,
        D2 = 2,
        kernLength1 = 5,
        kernLength2 = 5,
        pool  = 4,
        stridePool = 2,
        dropRate = 0.2,
        ELUAlpha = .1,
        batchMomentum = 0.25,
        seed = None
    ):
        _reset_seed(seed)
        super(EEGTokenizer, self).__init__()

        self.stridePool = stridePool
        self.pool = pool
        self.D1 = D1
        F1 = Chans*D1
        self.blck1 = nn.Sequential(
            nn.Conv1d(Chans, F1, kernLength1, padding = 'same', groups=Chans),
            nn.BatchNorm1d(F1, momentum = batchMomentum),
            nn.ELU(ELUAlpha),
        )
        
        self.pool1 = nn.AvgPool1d(pool, stridePool)
        self.drop1 = nn.Dropout1d(dropRate)

        self.blck2 = nn.Sequential(
            nn.Conv1d(F1, F1, kernLength2, padding = 'same', groups = F1),
            nn.BatchNorm1d(F1, momentum = batchMomentum),
            nn.ELU(ELUAlpha),
        )

        self.D2 = D2
        F2 = Chans*D1*D2
        self.blck3 = nn.Sequential(
            nn.Conv1d(F1, F2, kernLength2, padding = 'same', groups = F1),
            nn.BatchNorm1d(F2, momentum = batchMomentum),
            nn.ELU(ELUAlpha),
        )

        self.pool2 = nn.AvgPool1d(pool, stridePool)#, padding=1)
        self.drop2 = nn.Dropout1d(dropRate)

        self.blck4 = nn.Sequential(
            nn.Conv1d(F2, F2, kernLength2, padding = 'same', groups = F2),
            nn.BatchNorm1d(F2, momentum = batchMomentum),
            nn.ELU(ELUAlpha),
        )

    def forward(self, x):
        x1 = self.blck1(x)
        x1 = self.pool1(x1)
        x1 = self.drop1(x1)
        x2 = self.blck2(x1)
        x2 = x1 + x2 
        x3 = self.blck3(x2)
        x3 = self.pool2(x3)
        x3 = self.drop2(x3)
        x4 = self.blck4(x3)
        x4 = x3 + x4
        return x4


class TransformeegEncoder(nn.Module):

    def __init__(self, Chan, add_attention_pooling=False, seed=None):

        features = Chan*4
        self.features = features
        self.__do_attention_pooling = add_attention_pooling
        
        _reset_seed(seed)
        super(TransformeegEncoder, self).__init__()
        self.token_gen = EEGTokenizer(
            Chan,
            D1            = 2,      # 2
            D2            = 2,      # 2
            kernLength1   = 5,
            kernLength2   = 5,
            pool          = 4,       # 4
            stridePool    = 2,       # 2
            dropRate      = 0.2,
            ELUAlpha      = 0.1,
            batchMomentum = 0.25,
        )

        _reset_seed(seed)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                self.features,
                nhead                = 1,
                dim_feedforward      = self.features,
                dropout              = 0.2,
                activation           = torch.nn.functional.hardswish,
                batch_first          = True
            ),
            num_layers           = 2,
            enable_nested_tensor = False
        )
        #self.transformer.layers[1] = nn.TransformerEncoderLayer(
        #    self.features,
        #    nhead                = 1,
        #    dim_feedforward      = self.features,
        #    dropout              = 0.2,
        #    activation           = torch.nn.functional.hardswish,
        #    batch_first          = True
        #)
        if self.__do_attention_pooling:
            self.pool_lay = nn.MultiheadAttention(
                self.features,
                num_heads            = 1,
                dropout              = 0.2,
                batch_first          = True
            )
        else:
            self.pool_lay = torch.nn.AdaptiveAvgPool1d((1))

    def forward(self, x, mask=None):
        x = self.token_gen(x)
        x = torch.permute(x, [0, 2, 1])
        x = self.transformer(x, mask)
        if self.__do_attention_pooling:
            xq = torch.mean(x, 1, keepdim=True)
            x = self.pool_lay(xq, x, x)[0]
            x = torch.permute(x, [0, 2, 1])
        else:
            x = torch.permute(x, [0, 2, 1])
            x = self.pool_lay(x)
        x = x.squeeze(-1)
        return x


class TransformEEG(nn.Module):

    def __init__(
        self, nb_classes, Chan, Features,
        add_attention_pooling=False, freeze_encoder=-1, seed=None
    ):

        _reset_seed(seed)
        super(TransformEEG, self).__init__()
        self.Chan = Chan
        self.Features = Features
        self.freeze_encoder = freeze_encoder
        
        self.encoder = TransformeegEncoder(Chan, add_attention_pooling, seed)
        
        _reset_seed(seed)
        self.classification_head = nn.Sequential(
            nn.Linear(Features, Features//2 if Features//2>64 else 64),
            nn.LeakyReLU(),
            nn.Linear(Features//2 if Features//2>64 else 64, 1 if nb_classes <= 2 else nb_classes)
        )
        
    def forward(self, x):
        if self.freeze_encoder > 0:
            self.freeze_encoder -= 1
            self.encoder.requires_grad_(False)
        else:
            self.encoder.requires_grad_(True)
        x = self.encoder(x)
        x = self.classification_head(x)
        return x