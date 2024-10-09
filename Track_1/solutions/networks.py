from typing import List

import torch
import torch.nn.functional as F
from torch import nn


class PointNetFea(nn.Module):
    def __init__(self, point_dim, output_dim, batchnorm=False):
        super(PointNetFea, self).__init__()
        self.conv0 = nn.Conv1d(point_dim, 64, 1)
        self.conv1 = nn.Conv1d(64, 128, 1)
        self.conv2 = nn.Conv1d(128, 256, 1)
        self.conv3 = nn.Conv1d(256, 512, 1)
        if batchnorm:
            self.bn0 = nn.BatchNorm1d(64)
            self.bn1 = nn.BatchNorm1d(128)
            self.bn2 = nn.BatchNorm1d(256)
            self.bn3 = nn.BatchNorm1d(512)
        else:
            self.bn0 = nn.Identity()
            self.bn1 = nn.Identity()
            self.bn2 = nn.Identity()
            self.bn3 = nn.Identity()

        self.mlp1 = nn.Linear(512, 256)
        self.mlp2 = nn.Linear(256, output_dim)

    def forward(self, x):
        x = F.relu(self.bn0(self.conv0(x)))
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 512)
        x = F.relu(self.mlp1(x))
        x = self.mlp2(x)

        return x


class PointNetFeaNew(nn.Module):
    def __init__(self, point_dim, net_layers: List, batchnorm=False):
        super(PointNetFeaNew, self).__init__()
        self.layer_num = len(net_layers)
        self.conv0 = nn.Conv1d(point_dim, net_layers[0], 1)
        self.bn0 = nn.BatchNorm1d(net_layers[0]) if batchnorm else nn.Identity()
        for i in range(0, self.layer_num - 1):
            self.__setattr__(f"conv{i + 1}", nn.Conv1d(net_layers[i], net_layers[i + 1], 1))
            self.__setattr__(f"bn{i + 1}", nn.BatchNorm1d(net_layers[i + 1]) if batchnorm else nn.Identity())

        self.output_dim = net_layers[-1]

        # self.mlp1 = nn.Linear(512, 256)
        # self.mlp2 = nn.Linear(256, output_dim)

    def forward(self, x):
        for i in range(0, self.layer_num - 1):
            x = F.relu(self.__getattr__(f"bn{i}")(self.__getattr__(f"conv{i}")(x)))
        x = self.__getattr__(f"bn{self.layer_num - 1}")(self.__getattr__(f"conv{self.layer_num - 1}")(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, self.output_dim)
        # x = F.relu(self.mlp1(x))
        # x = self.mlp2(x)
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
        # self.mlp_feature_num = 256  # self.pointnet_global_feature_num  #256

        self.pointnet_local_fea = nn.Sequential(
            nn.Conv1d(dim, self.pointnet_local_feature_num, 1),
            nn.BatchNorm1d(self.pointnet_local_feature_num) if batchnorm else nn.Identity(),
            nn.ReLU(),
            nn.Conv1d(self.pointnet_local_feature_num, self.pointnet_local_feature_num, 1),
            nn.BatchNorm1d(self.pointnet_local_feature_num) if batchnorm else nn.Identity(),
            nn.ReLU(),
        )
        self.pointnet_global_fea = PointNetFeaNew(self.pointnet_local_feature_num, [64, 128, self.pointnet_global_feature_num], batchnorm=batchnorm)
        # self.pointnet_total_fea = PointNetFeaNew(
        #     self.pointnet_local_feature_num + self.pointnet_global_feature_num, [512, self.mlp_feature_num], batchnorm=batchnorm
        # )

        self.mlp_output = nn.Sequential(
            nn.Linear(self.pointnet_global_feature_num, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, out_dim)
        )

    def forward(self, marker_pos):
        """
        :param marker_pos: Tensor, size (batch, num_points, 4)
        :return:
        """
        if marker_pos.ndim == 2:
            marker_pos = torch.unsqueeze(marker_pos, dim=0)

        marker_pos = torch.transpose(marker_pos, 1, 2)

        batch_num = marker_pos.shape[0]
        point_num = marker_pos.shape[2]

        local_feature = self.pointnet_local_fea(marker_pos)  # (batch_num, self.pointnet_local_feature_num, point_num)
        # shape: (batch, step * 2, num_points)
        global_feature = self.pointnet_global_fea(local_feature).view(
            -1, self.pointnet_global_feature_num)  # (batch_num, self.pointnet_global_feature_num)

        # global_feature = global_feature.repeat(
        #     (1, 1, point_num)
        # )  # (batch_num, self.pointnet_global_feature_num, point_num)
        # combined_feature = torch.cat(
        #     [local_feature, global_feature], dim=1
        # )  # (batch_num, self.dim + self.pointnet_global_feature_num, point_num)

        # mlp_in = self.pointnet_total_fea(combined_feature)  # (batch_num, point_num, self.mlp_feature_num)
        pred = self.mlp_output(global_feature)
        # pred shape: (batch_num, out_dim)
        return pred
