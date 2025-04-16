import numpy as np
from torch import nn

def init_weights_(mod: nn.Module, gain: float = np.sqrt(2)):
    for m in mod.modules():
        if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            nn.init.xavier_normal_(m.weight, gain=gain)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m,
                        (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                         nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d,
                         nn.LayerNorm, nn.GroupNorm)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
    return mod
