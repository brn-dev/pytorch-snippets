from torch import nn


class DefaultInit(nn.Module):
    # copy this method into your module
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
                nn.init.xavier_normal_(m.weight, gain=2)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m,
                            (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                             nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d,
                             nn.LayerNorm, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
