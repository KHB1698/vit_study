import paddle
import paddle.nn as nn

# paddle.set_device('cpu')


class Identity(nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class Block(nn.Layer):
    def __init__(self, in_dim, out_dim, stride):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2D(in_channels=in_dim, out_channels=out_dim, kernel_size=3, stride=stride, padding=1, bias_attr=False)
        self.bn1 = nn.BatchNorm2D(out_dim)
        self.conv2 = nn.Conv2D(in_channels=out_dim, out_channels=out_dim, kernel_size=3, stride=1, padding=1, bias_attr=False)
        self.bn2 = nn.BatchNorm2D(out_dim)
        self.relu = nn.ReLU()

        if stride == 2 or in_dim != out_dim:
            self.downsample = nn.Sequential(
                nn.Conv2D(in_channels=in_dim, out_channels=out_dim, kernel_size=1, stride=stride, bias_attr=False),  # 用1*1卷积
                nn.BatchNorm2D(out_dim)
            )
        else:
            self.downsample = Identity()

    def forward(self, x):
        h = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        identity = self.downsample(h)
        x = x + identity
        x = self.relu(x)
        return x


class ResNet18(nn.Layer):
    def __init__(self, in_dim=64, num_classes=10):
        super(ResNet18, self).__init__()
        self.in_dim = in_dim
        # stem layers
        self.conv1 = nn.Conv2D(in_channels=3,
                               out_channels=in_dim,
                               kernel_size=3,  # 原本是7，为了训练该模型使用的3
                               stride=1,
                               padding=1,
                               bias_attr=False)
        self.bn1 = nn.BatchNorm2D(in_dim)
        self.relu = nn.ReLU()

        # blocks
        self.layers1 = self._make_layer(dim=64, n_blocks=2, stride=1)
        self.layers2 = self._make_layer(dim=128, n_blocks=2, stride=2)
        self.layers3 = self._make_layer(dim=256, n_blocks=2, stride=2)
        self.layers4 = self._make_layer(dim=512, n_blocks=2, stride=2)

        # head layer
        self.avgpool = nn.AdaptiveAvgPool2D(1)  # 根据我们想要的尺寸做pooling
        self.classifier = nn.Linear(512, num_classes)

    def _make_layer(self, dim, n_blocks, stride):
        layers_list = []
        layers_list.append(Block(self.in_dim, dim, stride))
        self.in_dim = dim
        for i in range(1, n_blocks):
            layers_list.append(Block(self.in_dim, dim, 1))
        return nn.Sequential(*layers_list)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layers1(x)
        x = self.layers2(x)
        x = self.layers3(x)
        x = self.layers4(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.classifier(x)

        return x


def main():
    t = paddle.randn([4, 3, 32, 32])
    model = ResNet18()
    out = model(t)
    print(out.shape)


if __name__ == '__main__':
    main()

#Out of memory error on GPU 0. Cannot allocate 128.000244MB memory on GPU 0, 1.992853GB memory has been allocated and available memory is only 7.193556MB.