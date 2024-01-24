from .vit import VisionTransformer, CONFIGS
from .data import Get_Data
from .resnet import Resnet
from .resnet import residualunit
import torch.nn.functional as F
import torch.nn as nn
import torch



configs = CONFIGS['ViT-B_16']
IMG_SIZE = 256
NUM_CLASSES = 45
LAYERS = [3, 4, 6, 3]


class CenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, device):
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim, device=device))

    def forward(self, x, labels):
        centers_batch = self.centers.index_select(0, labels)
        loss = (x - centers_batch).pow(2).sum() / 2.0
        return loss


class WeightedFeatures(nn.Module):
    def __init__(self, feature_size):
        super().__init__()
        self.weights = nn.Linear(2 * feature_size, 2)

    def forward(self, A, B):
        combined_features = torch.cat((A, B), dim=1)
        weights = torch.softmax(self.weights(combined_features), dim=1)
        # return torch.cat((weights[:, 0:1] * A, weights[:, 1:2] * B), dim=1)
        return weights[:, 0:1] * A + weights[:, 1:2] * B


class CTNet(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.ViT = VisionTransformer(configs, IMG_SIZE, NUM_CLASSES)
        self.Res = Resnet(residualunit, LAYERS)
        self.criterion = nn.CrossEntropyLoss(reduction='sum')
        self.feature_weight = WeightedFeatures(512)

        self.conv1 = nn.Linear(768, 512)
        self.softmax = nn.Linear(512, 45)
        self.batchnorm = nn.BatchNorm1d(512)
        self.dropout = nn.Dropout(p=0.2)
        self.relu = nn.ReLU(inplace=True)
        

    def forward(self, x, labels):
        vit_features, output_1 = self.ViT(x)
        criterion1 = self.criterion(output_1, labels)
        vit_features = self.conv1(vit_features)
        centerloss1 = CenterLoss(NUM_CLASSES, 512, self.device)
        loss1 = centerloss1(vit_features, labels)

        res_features, output_2 = self.Res(x)
        criterion2 = self.criterion(output_2, labels)
        centerloss2 = CenterLoss(NUM_CLASSES, 512, self.device)
        loss2 = centerloss2(res_features, labels)

        features = self.feature_weight(vit_features, res_features)
        features = self.batchnorm(features)
        features = self.relu(features)
        features = self.dropout(features)
        output_3 = self.softmax(features)
        criterion3 = self.criterion(output_3, labels)

        total_loss = criterion1+criterion2+criterion3+(loss1+loss2)*0.005

        return output_3, total_loss
