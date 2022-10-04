import paddle
import paddle.nn as nn
# paddle.vision.models.resnet18()

#paddle.set_device('cpu')


class Block(nn.Layer):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv1 = nn.Conv2D(in_channels, out_channels,
                               kernel_size=3, padding=1, stride=stride, bias_attr=False)
        self.bn1 = nn.BatchNorm2D(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2D(out_channels, out_channels,
                               kernel_size=3, padding=1, bias_attr=False)
        self.bn2 = nn.BatchNorm2D(out_channels)

        self.downsample = None
        if in_channels != out_channels and stride == 2:
            self.downsample = nn.Conv2D(in_channels, out_channels,
                                        kernel_size=1, stride=stride)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        return self.relu(out + identity)

class ResNet18(nn.Layer):
    def __init__(self, base_channels=64, num_classes=1000):
        super().__init__()
        self.conv1 = nn.Conv2D(3, base_channels, kernel_size=7, padding=3, bias_attr=False)
        self.maxpool = nn.MaxPool2D(kernel_size=3, padding=1, stride=2)

        self.layer1 = self._make_layer(base_channels, base_channels, 2, stride=1)
        self.layer2 = self._make_layer(base_channels, base_channels * 2, 2, stride=2)
        self.layer3 = self._make_layer(base_channels * 2, base_channels * 4, 2, stride=2)
        self.layer4 = self._make_layer(base_channels * 4, base_channels * 8, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2D((1, 1))

        self.fc = nn.Linear(base_channels * 8, num_classes, bias_attr=False)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        block_list = list()
        block_list.append(
            Block(in_channels, out_channels, stride=stride)
        )
        for i in range(1, num_blocks):
            block_list.append(
                Block(out_channels, out_channels, stride=1)
            )

        return nn.Sequential(*block_list)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = paddle.flatten(x, start_axis=1)
        x = self.fc(x)

        return x


def main():
    model = ResNet18(num_classes=10)
    print(model)
    x = paddle.randn([2, 3, 32, 32])
    out = model(x)
    print(out.shape)


if __name__ == "__main__":
    main()
