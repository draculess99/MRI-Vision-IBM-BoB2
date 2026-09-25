"""Twelve-target adapter around the shared V0.5 MRNet encoder."""

from .cnn_model import MRNetCNN

RSNA_TARGET_COUNT = 12


class RSNAKneeCNN(MRNetCNN):
    def __init__(self, num_planes=3, dropout=0.2):
        super().__init__(num_planes=num_planes, dropout=dropout, num_targets=RSNA_TARGET_COUNT)
