import torch
import torch.nn as nn
import torch.nn.functional as F


def get_norm_3d(norm_type, num_features):
    """
    简单的 3D 归一化工厂函数：
    norm_type: 'instance' 或 'batch' 或 None
    """
    if norm_type is None:
        return nn.Identity()
    if norm_type.lower() in ['in', 'instance', 'instancenorm']:
        return nn.InstanceNorm3d(num_features, affine=True)
    if norm_type.lower() in ['bn', 'batch', 'batchnorm']:
        return nn.BatchNorm3d(num_features)
    raise ValueError(f"Unsupported norm type: {norm_type}")


class ConvBlock3D(nn.Module):
    """
    基础 3D 卷积块：Conv3d + Norm + LeakyReLU
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 norm_type='instance',
                 negative_slope=0.01):
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=(norm_type is None)
        )
        self.norm = get_norm_3d(norm_type, out_channels)
        self.act = nn.LeakyReLU(negative_slope=negative_slope, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class FDMDS(nn.Module):

    def __init__(self,
                 in_channels=4,
                 mid_channels=16,
                 norm_type='instance',
                 negative_slope=0.01):
        super().__init__()

        # 第一层：升维到 mid_channels
        self.conv1 = ConvBlock3D(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            norm_type=norm_type,
            negative_slope=negative_slope
        )

        # 第二层：在 mid_channels 维度上再堆一层
        self.conv2 = ConvBlock3D(
            in_channels=mid_channels,
            out_channels=mid_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            norm_type=norm_type,
            negative_slope=negative_slope
        )

        # 第三层：降回原始通道数（例如 4 模态）
        self.conv3 = nn.Conv3d(
            in_channels=mid_channels,
            out_channels=in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True  # 最后一层可以带 bias
        )

        # 可选：再加一个非线性，让输出稍微“收一收”
        self.out_act = nn.LeakyReLU(negative_slope=negative_slope, inplace=True)

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """
        Kaiming 初始化，适配 LeakyReLU
        """
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, a=0.01, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, (nn.BatchNorm3d, nn.InstanceNorm3d)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        """
        x: [B, C_in, D, H, W]  模糊四模态体
        返回: [B, C_in, D, H, W] 去模糊后的体
        """
        identity = x  # 用残差保证不会让网络一上来改得太离谱

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        # 残差连接：鼓励网络学习“残差模糊成分”
        out = out + identity

        # 轻微非线性收一下
        out = self.out_act(out)

        return out
if __name__ == "__main__":
    stem = FDMDS(in_channels=4, mid_channels=16, norm_type='instance')
    x = torch.randn(1, 4, 96, 128, 128)  # 一块 3D patch
    y = stem(x)
    print(y.shape)  # 期待输出: torch.Size([1, 4, 96, 128, 128])
