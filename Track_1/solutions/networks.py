from typing import List

import torch
import torch.nn.functional as F
from torch import nn


class PointNetFeaNew(nn.Module):
    def __init__(self, point_dim, net_layers: List, batchnorm=False):
        super(PointNetFeaNew, self).__init__()
        self.layer_num = len(net_layers)
        self.conv0 = nn.Conv1d(point_dim, net_layers[0], 1)
        self.bn0 = nn.BatchNorm1d(net_layers[0]) if batchnorm else nn.Identity()
        for i in range(0, self.layer_num - 1):
            self.__setattr__(
                f"conv{i + 1}", nn.Conv1d(net_layers[i], net_layers[i + 1], 1)
            )
            self.__setattr__(
                f"bn{i + 1}",
                nn.BatchNorm1d(net_layers[i + 1]) if batchnorm else nn.Identity(),
            )

        self.output_dim = net_layers[-1]

    def forward(self, x):
        for i in range(0, self.layer_num - 1):
            x = F.relu(self.__getattr__(f"bn{i}")(self.__getattr__(f"conv{i}")(x)))
        x = self.__getattr__(f"bn{self.layer_num - 1}")(
            self.__getattr__(f"conv{self.layer_num - 1}")(x)
        )
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, self.output_dim)
        return x


class PointNetFeatureExtractor(nn.Module):
    """
    this is a latent feature extractor for point cloud data
    need to distinguish this from other modules defined in feature_extractors.py
    those modules are only used to extract the corresponding input (e.g. point flow, manual feature, etc.) from original observations
    """

    def __init__(self, dim, out_dim, batchnorm=False):
        super(PointNetFeatureExtractor, self).__init__()
        self.dim = dim

        self.pointnet_local_feature_num = 64
        self.pointnet_global_feature_num = 512

        self.pointnet_local_fea = nn.Sequential(
            nn.Conv1d(dim, self.pointnet_local_feature_num, 1),
            (
                nn.BatchNorm1d(self.pointnet_local_feature_num)
                if batchnorm
                else nn.Identity()
            ),
            nn.ReLU(),
            nn.Conv1d(
                self.pointnet_local_feature_num, self.pointnet_local_feature_num, 1
            ),
            (
                nn.BatchNorm1d(self.pointnet_local_feature_num)
                if batchnorm
                else nn.Identity()
            ),
            nn.ReLU(),
        )
        self.pointnet_global_fea = PointNetFeaNew(
            self.pointnet_local_feature_num,
            [64, 128, self.pointnet_global_feature_num],
            batchnorm=batchnorm,
        )

        self.mlp_output = nn.Sequential(
            nn.Linear(self.pointnet_global_feature_num, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, out_dim),
        )

    def forward(self, marker_pos):
        """
        :param marker_pos: Tensor, size (batch, num_points, 4)
        :return:
        """
        if marker_pos.ndim == 2:
            marker_pos = torch.unsqueeze(marker_pos, dim=0)

        marker_pos = torch.transpose(marker_pos, 1, 2)
        local_feature = self.pointnet_local_fea(
            marker_pos
        )  # (batch_num, self.pointnet_local_feature_num, point_num)
        # shape: (batch, step * 2, num_points)
        global_feature = self.pointnet_global_fea(local_feature).view(
            -1, self.pointnet_global_feature_num
        )  # (batch_num, self.pointnet_global_feature_num)

        pred = self.mlp_output(global_feature)
        # pred shape: (batch_num, out_dim)
        return pred
