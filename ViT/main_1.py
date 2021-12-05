import paddle
import paddle.nn as nn

paddle.set_device('cpu')


class Identity(nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class Mlp(nn.Layer):
    def __init__(self, embed_dim, mlp_ratio=4.0, dropout=0.):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, int(embed_dim * mlp_ratio))  # 增大mlp_ratio倍
        self.fc2 = nn.Linear(int(embed_dim * mlp_ratio), embed_dim)  # 回到原来的维度
        self.act = nn.GELU()  # 激活函数，gelu与relu的区别在于激活函数的输出范围
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class PatchEmbedding(nn.Layer):
    '''
    image_size 图片的大小
    patch_size 卷积核大小,每个patch的大小
    in_channels 图片的通道数
    embed_dim 卷积核的维度
    dropout 卷积核的dropout    
    '''

    def __init__(self, image_size, patch_size, in_channels, embed_dim, dropout=0.):
        super().__init__()
        self.patch_embedding = nn.Conv2D(in_channels,
                                         embed_dim,
                                         kernel_size=patch_size,
                                         stride=patch_size,
                                         weight_attr=paddle.ParamAttr(initializer=nn.initializer.Constant(value=1.0)),
                                         bias_attr=False)
        self.dropout = nn.Dropout(dropout)
        
        

    def forward(self, x):
        # [n, c, h, w]
        x = self.patch_embedding(x)  # [n, c', h', w']
        x = x.flatten(2)  # [n, c', h'*w']
        x = x.transpose([0, 2, 1])  # [n, h'*w', c']
        x = self.dropout(x)
        return x


class Attention(nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class EncoderLayer(nn.Layer):
    def __init__(self, embed_dim):
        super().__init__()
        self.attn_norm = nn.LayerNorm(embed_dim)
        self.attn = Attention()
        self.mlp_norm = nn.LayerNorm(embed_dim)
        self.mlp = Mlp(embed_dim)

    def forward(self, x):
        h = x
        x = self.attn_norm(x)
        x = self.attn(x)
        x = x + h

        h = x
        x = self.mlp_norm(x)
        x = self.mlp(x)
        x = x + h
        return x


class ViT(nn.Layer):
    def __init__(self):
        super().__init__()
        self.patch_embed = PatchEmbedding(224, 7, 3, 16)
        layer_list = [EncoderLayer(16) for i in range(5)]  # 5个encoder
        self.encoders = nn.LayerList(layer_list)  # 层列表，也可用nn.sequential()实现
        self.head = nn.Linear(16, 10)
        self.avgpool = nn.AdaptiveAvgPool1D(1)  # 经过encoder的所有数取一个平均
        self.norm = nn.LayerNorm(16)

    def forward(self, x):
        # [n, 3, 224, 224]做7*7的卷积->[n, 16, 32, 32]->[n,1024,16]有1024个token
        x = self.patch_embed(x)  # [n, h*w, c]: 4, 1024, 16
        for encoder in self.encoders:
            x = encoder(x)  # [n, h*w, c]输入和输出维度不变
        # avg
        x = self.norm(x)  # 1024个token，每个token的维度为16来做layerNorm,形状不变
        x = x.transpose([0, 2, 1])  # 转置成[n, c, 1024] n,16,1024
        x = self.avgpool(x)  # [n, c, 1]，1024做平均池化，应该合在一起代表一整张图来做分类
        x = x.flatten(1)  # [n, c]
        x = self.head(x)
        return x


def main():
    vit = ViT()
    print(vit)
    paddle.summary(vit, input_size=(4, 3, 224, 224))


if __name__ == "__main__":
    main()
