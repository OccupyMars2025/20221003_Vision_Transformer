import paddle
import paddle.nn as nn

#paddle.set_device('cpu')

class Identity(nn.Layer):
    def __init_(self):
        super().__init__()

    def forward(self, x):
        return x


class Block(nn.Layer):
    def __init__(self, in_dim, out_dim, stride):
        super().__init__()
        ## 补充代码

    def forward(self, x):
        ## 补充代码


class ResNet18(nn.Layer):
    def __init__(self, in_dim=64, num_classes=1000):
        super().__init__()
        ## 补充代码

    def _make_layer(self, out_dim, n_blocks, stride):
        ## 补充代码

    def forward(self, x):
        ## 补充代码
         

def main():
    model = ResNet18()
    print(model)
    x = paddle.randn([2, 3, 32, 32])
    out = model(x)
    print(out.shape)

if __name__ == "__main__":
    main()
