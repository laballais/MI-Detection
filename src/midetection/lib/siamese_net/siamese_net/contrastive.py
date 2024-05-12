import torch
import numpy as np
import torch.nn.functional as F
from midetection.lib.siamese_net.siamese_net import config
import torch.nn as nn


class ContrastiveLoss(torch.nn.Module):
    "Contrastive loss function"

    def __init__(self, margin=config.margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin        # min distance that LV wall needs to keep for non-MI case. if dist < margin = MI. if dist > margin = nonMI
        self.dist = []

    # distance for similar and another margin-distance for dissimilar
    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        # loss_contrastive = torch.mean(
        #     (1 - label) * torch.pow(euclidean_distance, 2)
        #     + (label)
        #     * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        # )

        # # min distance that LV wall needs to keep for non-MI case. if dist < margin = nonMI. if dist > margin = MI
        loss_contrastive = torch.mean(
            (label) * torch.pow(euclidean_distance, 2)
            + (1 - label)
            * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        )
        # self.dist.append(euclidean_distance.detach().tolist())

        return loss_contrastive
