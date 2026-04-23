import torch
import torch.nn as nn
from torchvision import models
from torchvision import transforms
import torchvision.transforms.functional as F

from PIL import Image, ImageFilter

class VideoTransformer(nn.Module):
    def __init__(self, num_classes):
        super(VideoTransformer, self).__init__()
        # 1. バックボーン（画像から特徴抽出）
        # 例: ResNet18の最終層以外を利用 (特徴量サイズ 512)
        #resnet = models.resnet18(pretrained=True)
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        # 2. Transformer層
        # d_model: 特徴量次元, nhead: 注意機構のヘッド数
        encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)

        # 3. 分類用ヘッド
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        # xの形状: (Batch, Time, Channel, Height, Width)
        b, t, c, h, w = x.shape

        # 各フレームをバックボーンに通す
        # (B*T, C, H, W) に変換して一括処理
        x = x.view(b * t, c, h, w)
        features = self.backbone(x) # (B*T, 512, 1, 1)
        features = features.view(b, t, -1) # (Batch, Time, 512)

        # Transformerは (Sequence, Batch, Feature) の形状を期待する
        features = features.permute(1, 0, 2)

        # Transformerによる時系列処理
        out = self.transformer(features) # (Time, Batch, 512)

        # 最後のフレームの出力、または平均プーリングを使用して分類
        out = out.mean(dim=0) # (Batch, 512)
        return self.fc(out)

