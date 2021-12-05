import paddle
import paddle.nn as nn
paddle.set_device('cpu')


class Attention(nn.Layer):
    def __init__(self, embed_dim, num_heads, qkv_bias=False, qk_scale=None, dropout=0.0, attention_dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads  # 因为我们要保持维度一致,所以每个head的维度都是embed_dim/num_heads
        self.all_head_dim = self.head_dim * num_heads  # 所有head的维度,为了整除定义的变量
        self.qkv = nn.Linear(self.embed_dim,
                             self.all_head_dim * 3,  # qkv拼起来的维度 = embed_dim * 3
                             bias_attr=False if qkv_bias is False else None  # 如果qkv_bias为False,则不使用bias,否则为None（None）其实是有初始值的，默认为0
                             )
        self.scale = self.head_dim ** -0.5 if qk_scale is None else qk_scale  # 避免人为设置
        self.softmax = nn.Softmax(-1)
        self.proj = nn.Linear(self.all_head_dim, embed_dim)

    def transpose_multi_head(self, x):
        # x:[B, N, all_head_dim]
        new_shape = x.shape[:-1] + [self.num_heads, self.head_dim] # 记住方法，直接拼接
        # new_shape=[B, N, num_heads, head_dim]
        x = x.reshape(new_shape)
        # x:[B, N, num_heads, head_dim]
        x = x.transpose([0, 2, 1, 3])
        # x:[B, num_heads, N, head_dim],N为num_patchs
        return x

    def forward(self, x):
        B, N, _ = x.shape
        qkv = self.qkv(x).chunk(3, -1) # [B, N, all_head_dim] -> [B, N, all_head_dim*3] -> 划分成3个部分[B, N, all_head_dim]
        # [B, N, all_head_dim]*3
        q, k, v = map(self.transpose_multi_head, qkv)
        # q,k,v:[B, num_heads, N, head_dim]
        attn = paddle.matmul(q, k, transpose_y=True)  # q乘以k的转置，即qk的点乘
        attn = self.scale * attn
        attn = self.softmax(attn)
        attn_weight = attn
        # 可以定义dropout
        # attn: [B, num_heads, num_patchs, num_patchs]
        out = paddle.matmul(attn, v)  # 将attn乘以v，即attention的输出,softmax(scale*(q*k'))*v
        out = out.transpose([0, 2, 1, 3])
        # attn: [B, num_patchs, num_heads, head_dim]
        out = out.reshape([B, N, -1])
        out = self.proj(out)
        # dropout
        return out, attn_weight


def main():
    t = paddle.randn([8, 16, 96])
    model = Attention(embed_dim=96, num_heads=4, qkv_bias=False, qk_scale=None)
    # print(model)
    # out, w = model(t)
    # print(out.shape)
    # print(w.shape)
    
    params_info = paddle.summary(model, (8,16,96))
    print(params_info)


if __name__ == '__main__':
    main()
