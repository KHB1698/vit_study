import paddle
import paddle.nn as nn
from mask import generate_mask
paddle.set_device('cpu')


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
        # [B, H, W, C]
        x = x.reshape([b, h, w, c])
        # 实现Merge, 进行数据拆分, 得到多个数据Shape: [B, H//2, W//2, C]
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 0::2, 1::2, :]
        x2 = x[:, 1::2, 0::2, :]
        x3 = x[:, 1::2, 1::2, :]
        # 得到新的x数据:拼接拆分的数据得到Shape: [B, H//2, W//2, 4C]
        x = paddle.concat([x0, x1, x2, x3], axis=-1)
        # [B, (H//2)*(W//2), 4C]
        x = x.reshape([b, -1, 4*c])
        # 利用已有的norm和线性层实现最后的Merge映射, 注意这里为PreNorm(先归一化哦)
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
    # B, H/ws, ws, W/ws, ws, C
    x = x.reshape([B, H//window_size, window_size, W//window_size, window_size, C])
    # B, H/ws, W/ws, ws, ws, c
    x = x.transpose([0, 1, 3, 2, 4, 5])
    # B * H/ws * W/ws, ws, ws, c
    x = x.reshape([-1, window_size, window_size, C])
    # x = x.reahspe([-1, window_size*window_size, C]) # [B*num_windows, ws*ws, C]
    return x


# windows_partition反过程，把窗口拼接起来
def windows_reverse(windows, window_size, H, W):
    # windows: [B*num_windows, ws*ws, C]
    B = int(windows.shape[0] // (H/window_size * W/window_size))
    x = windows.reshape([B, H//window_size, W//window_size, window_size, window_size, -1])
    x = x.transpose([0, 1, 3, 2, 4, 5])  # [B, H/ws, ws, W/ws, ws, C]
    x = x.reshape([B, H, W, -1])  # [B, H, W, C]
    return x


class WindowAttention(nn.Layer):
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim = dim
        self.dim_head = dim // num_heads
        self.num_heads = num_heads
        self.scale = self.dim_head ** -0.5
        self.softmax = nn.Softmax(-1)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

        # Relative Position Bias相对位置偏差
        self.window_size = window_size
        self.relative_position_bias_table = paddle.create_parameter(
            shape=[(2*window_size-1)*(2*window_size-1), num_heads],
            dtype='float32',
            default_initializer=nn.initializer.TruncatedNormal(std=.02))
        coord_h = paddle.arange(self.window_size)
        coord_w = paddle.arange(self.window_size)
        # ! 注意方法
        coords = paddle.stack(paddle.meshgrid([coord_h, coord_w]))  # [2, ws, ws]
        coords = coords.flatten(1)  # [2, ws*ws]
        relative_coords = coords.unsqueeze(2) - coords.unsqueeze(1) # [2, ws*ws, ws*ws]列减行
        relative_coords = relative_coords.transpose([1, 2, 0])
        relative_coords[:, :, 0] += self.window_size - 1
        relative_coords[:, :, 1] += self.window_size - 1

        relative_coords[:, :, 0] *= 2*self.window_size - 1
        relative_coords_index = relative_coords.sum(2)
        print(relative_coords_index)
        # register_buffer(name, value)将变量的值保存到内存中，可以通过name来访问,但不会被训练
        self.register_buffer('relative_coords_index', relative_coords_index)

    # Relative Position Bias
    def get_relative_position_bias_from_index(self):
        table = self.relative_position_bias_table  # [2m-1 * 2m-1, num_heads]
        print('table shape=', table.shape)
        # 已经加入缓冲区了所以可以self直接访问
        index = self.relative_coords_index.reshape([-1])  # [M^2, M^2] - > [M^2*M^2]
        print('index shape =', index.shape)
        relative_position_bias = paddle.index_select(x=table, index=index)  # [M*M, M*M, num_heads]
        return relative_position_bias

    def transpose_multi_head(self, x):
        new_shape = x.shape[:-1] + [self.num_heads, self.dim_head]
        x = x.reshape(new_shape)
        x = x.transpose([0, 2, 1, 3])  # [B, num_heads, num_patches, dim_head]
        return x

    def forward(self, x, mask=None):
        # x: [B*num_windows, ws*ws, c]
        B, N, C = x.shape
        print('xshape=', x.shape)
        qkv = self.qkv(x).chunk(3, axis=-1)
        q, k, v = map(self.transpose_multi_head, qkv)
        q = q * self.scale
        attn = paddle.matmul(q, k, transpose_y=True)
        # [B*num_windows, num_heads, num_patches, num_patches]  num_patches = windows_size * window_size = M * M

        print('attn shape=', attn.shape)
        # BEGIN Class 6: Relative Position Bias
        relative_position_bias = self.get_relative_position_bias_from_index()
        relative_position_bias = relative_position_bias.reshape([self.window_size * self.window_size, self.window_size * self.window_size, -1])
        # [M*M, M*M, num_heads]
        relative_position_bias = relative_position_bias.transpose([2, 0, 1])  # [num_heads, M*M, M*M]
        attn = attn + relative_position_bias.unsqueeze(0)
        # END Class 6: Relative Position Bias
        print('attn shape=', attn.shape)

        if mask is None:
            attn = self.softmax(attn)
        else:
            # mask: [num_windows, num_patches, num_patches]
            # attn: [B*num_windows, num_heads, num_patches, num_patches]
            # 将mask与attn相加，所以attn要reshape->[B, num_windows, num_heads, num_patches, num_patches]
            attn = attn.reshape([x.shape[0]//mask.shape[0], mask.shape[0], self.num_heads, mask.shape[1], mask.shape[1]])
            # mask: [1, num_windows, 1, num_patches, num_patches]与attn相对应再相加
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            # attn还原
            attn = attn.reshape([-1, self.num_heads, mask.shape[1], mask.shape[1]])
            attn = self.softmax(attn)

        out = paddle.matmul(attn, v)
        out = out.transpose([0, 2, 1, 3])
        out = out.reshape([B, N, C])
        out = self.proj(out)
        return out


class SwinBlock(nn.Layer):
    def __init__(self, dim, input_resolution, num_heads, window_size, shift_size=0):
        super().__init__()
        self.dim = dim
        self.resolution = input_resolution
        self.window_size = window_size
        self.shift_size = shift_size
        self.attn_norm = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads)

        self.mlp_norm = nn.LayerNorm(dim)
        self.mlp = Mlp(dim)

        # generate mask and register buffer
        if self.shift_size > 0:
            attn_mask = generate_mask(self.window_size, self.shift_size, input_resolution=self.resolution)
        else:
            attn_mask = None
        self.register_buffer('attn_mask', attn_mask)

    def forward(self, x):
        H, W = self.resolution
        B, N, C = x.shape

        h = x
        x = self.attn_norm(x)

        x = x.reshape([B, H, W, C])
        # shift window
        if self.shift_size > 0:
            shifted_x = paddle.roll(x, shifts=(-self.shift_size, -self.shift_size), axis=(1, 2))
        else:
            shifted_x = x
        # compute window attn
        x_windows = windows_partition(shifted_x, self.window_size)
        x_windows = x_windows.reshape([-1, self.window_size*self.window_size, C])
        attn_windows = self.attn(x_windows, mask=self.attn_mask)
        attn_windows = attn_windows.reshape([-1, self.window_size, self.window_size, C])
        shifted_x = windows_reverse(attn_windows, self.window_size, H, W)
        # shift back
        if self.shift_size > 0:
            x = paddle.roll(shifted_x, shifts=(self.shift_size, self.shift_size), axis=(1, 2))
        else:
            x = shifted_x
        # [B, H, W, C]
        x = x.reshape([B, H*W, C])
        x = x + h
        h = x
        x = self.mlp_norm(x)
        x = self.mlp(x)
        x = x + h
        return x


def main():
    t = paddle.randn([4, 3, 224, 224])
    patch_embedding = PatchEmbedding(patch_size=4, embed_dim=96)
    swin_block_w_msa = SwinBlock(dim=96, input_resolution=[56, 56], num_heads=4, window_size=7, shift_size=0)
    swin_block_sw_msa = SwinBlock(dim=96, input_resolution=[56, 56], num_heads=4, window_size=7, shift_size=7//2)
    patch_merging = PatchMerging(input_resolution=[56, 56], dim=96)

    print('image shape=[4, 3, 224, 224]')
    out = patch_embedding(t)  # [4, 56,56,96]
    print('patch embedding out shape:', out.shape)
    out = swin_block_w_msa(out)  # [4, 56,56,96]
    out = swin_block_sw_msa(out)  # [4, 56,56,96]
    print('swin block out shape:', out.shape)
    out = patch_merging(out)  # [4, 3, 224, 224]
    print('patch merging out shape:', out.shape)


if __name__ == '__main__':
    main()
