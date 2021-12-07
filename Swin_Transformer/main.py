import paddle
import paddle.nn as nn


class PatchEmbedding(nn.Layer):
    def __init__(self, patch_size=4, embed_dim=96):
        super().__init__()
        self.patch_embed = nn.Conv2D(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.patch_embed(x)  # [n, embed_dim, h', w']
        x = x.flatten(2)  # [n, embed_dim, h'*w']
        x = x.transpose([0, 2, 1])  # [n, h'*w', embed_dim]
        x = self.norm(x)
        return x


class PatchMerging(nn.Layer):
    def __init__(self, input_resolution, dim):
        super().__init__()
        self.resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4*dim, 2*dim)
        self.norm = nn.LayerNorm(4*dim)

    def forward(self, x):
        h, w = self.resolution
        b, _, c = x.shape
        x = x.reshape([b, h, w, c])

        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 0::2, 1::2, :]
        x2 = x[:, 1::2, 0::2, :]
        x3 = x[:, 1::2, 1::2, :]

        x = paddle.concat([x0, x1, x2, x3], axis=-1)  # [b, h/2, w/2, 4c]
        x = x.reshape([b, -1, 4*c])
        x = self.norm(x)
        x = self.reduction(x)
        return x


class Mlp(nn.Layer):
    def __init__(self, dim, mlp_ratio=4.0, dropout=0.):
        super().__init__()
        self.fc1 = nn.Linear(dim, int(dim * mlp_ratio))
        self.fc2 = nn.Linear(int(dim * mlp_ratio), dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


# windows_partition不需要学习，所以定义一个方法就行
def windows_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.reshape([B, H//window_size, window_size, W//window_size, window_size, C])
    x = x.transpose([0, 1, 3, 2, 4, 5])
    # [B, H/window_size, W/window_size, window_size, window_size, C]
    x = x.reshape([[-1, window_size, window_size, C]])  # ? 注意这里的reshape，两个中括号
    # [B*H/window_size*W/window_size, window_size, window_size, C]
    return x


# windows_partition反过程，把窗口拼接起来
def windows_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] // (H/window_size * W/window_size))
    x = windows.reshape([B, H//window_size, W//window_size, window_size, window_size, -1])
    x = x.transpose([0, 1, 3, 2, 4, 5])
    x = x.reshape([B, H, W, -1])
    return x
