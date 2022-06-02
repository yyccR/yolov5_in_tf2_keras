import torch
import torch.nn as nn
import math
import tensorflow as tf


# class Conv(nn.Module):
#     # Standard convolution
#     def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
#         super().__init__()
#         self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
#         self.bn = nn.BatchNorm2d(c2)
#         self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
#
#     def forward(self, x):
#         return self.act(self.bn(self.conv(x)))
#
#     def forward_fuse(self, x):
#         return self.act(self.conv(x))

class Conv(tf.keras.layers.Layer):
    def __init__(self, out_channels, kernel_size=1, stride=1, padding='same', groups=1, act=True):
        """ 卷积计算, 2d卷积->bn->swish(silu)
        :param features:
        :param out_channels:
        :param kernel_size:
        :param stride:
        :param padding:
        :param groups: 分组卷积参数
        :param act: 是否使用激活函数
        :return:
        """
        super(Conv, self).__init__()
        self.act = act
        self.conv = tf.keras.layers.Conv2D(
            filters=out_channels,
            kernel_size=kernel_size,
            strides=stride,
            padding=padding,
            groups=groups,
            use_bias=False
        )
        self.bn = tf.keras.layers.BatchNormalization(momentum=0.75)
        self.relu = tf.keras.layers.ReLU()

    def call(self, inputs, training=None, *args, **kwargs):
        x = self.conv(inputs)
        x = self.bn(x)
        if self.act:
            # x = tf.keras.activations.swish(x)
            x = self.relu(x)
        return x


#
# def Conv(features, out_channels, kernel_size=1, stride=1, padding=None, groups=1, act=True):
#     x = tf.keras.layers.Conv2D(
#         filters=out_channels,
#         kernel_size=kernel_size,
#         strides=stride,
#         padding='same',
#         groups=groups,
#         use_bias=False
#     )(features)
#     x = tf.keras.layers.BatchNormalization()(x)
#     if act:
#         x = tf.keras.activations.swish(x)
#     return x


# class DWConv(Conv):
#     # Depth-wise convolution class
#     def __init__(self, c1, c2, k=1, s=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
#         super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), act=act)


# def DWConv(features, out_channels, kernel_size=1, stride=1, act=True):
#     in_channels = tf.shape(features)[-1]
#
#     groups = math.gcd(in_channels, out_channels)
#     output = Conv(features=features,
#                   out_channels=out_channels,
#                   kernel_size=kernel_size,
#                   stride=stride,
#                   groups=groups,
#                   act=act)
#     return output


class DWConv(tf.keras.layers.Layer):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, act=True):
        super(DWConv, self).__init__()
        # 求最大公约, 分组卷积需要考虑到的最大分组
        groups = math.gcd(in_channels, out_channels)
        self.conv = Conv(out_channels=out_channels, kernel_size=kernel_size, stride=stride, groups=groups, act=act)

    def call(self, inputs, *args, **kwargs):
        return self.conv(inputs)


# class TransformerLayer(nn.Module):
#     # Transformer layer https://arxiv.org/abs/2010.11929 (LayerNorm layers removed for better performance)
#     def __init__(self, c, num_heads):
#         super().__init__()
#         self.q = nn.Linear(c, c, bias=False)
#         self.k = nn.Linear(c, c, bias=False)
#         self.v = nn.Linear(c, c, bias=False)
#         self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)
#         self.fc1 = nn.Linear(c, c, bias=False)
#         self.fc2 = nn.Linear(c, c, bias=False)
#
#     def forward(self, x):
#         x = self.ma(self.q(x), self.k(x), self.v(x))[0] + x
#         x = self.fc2(self.fc1(x)) + x
#         return x


class TransformerLayer(tf.keras.layers.Layer):
    # Transformer layer https://arxiv.org/abs/2010.11929 (LayerNorm layers removed for better performance)
    def __init__(self, c, num_heads):
        super(TransformerLayer, self).__init__()
        self.q = tf.keras.layers.Dense(c, use_bias=False)
        self.k = tf.keras.layers.Dense(c, use_bias=False)
        self.v = tf.keras.layers.Dense(c, use_bias=False)
        self.multiheadAttention = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=c, value_dim=c)
        self.fc1 = tf.keras.layers.Dense(c, use_bias=False)
        self.fc2 = tf.keras.layers.Dense(c, use_bias=False)

    def call(self, inputs, *args, **kwargs):
        y = self.multiheadAttention(self.q(inputs), self.v(inputs), self.k(inputs)) + inputs
        x = self.fc1(y)
        x = self.fc2(x)
        x = x + y
        return x


# class TransformerBlock(nn.Module):
#     # Vision Transformer https://arxiv.org/abs/2010.11929
#     def __init__(self, c1, c2, num_heads, num_layers):
#         super().__init__()
#         self.conv = None
#         if c1 != c2:
#             self.conv = Conv(c1, c2)
#         self.linear = nn.Linear(c2, c2)  # learnable position embedding
#         self.tr = nn.Sequential(*[TransformerLayer(c2, num_heads) for _ in range(num_layers)])
#         self.c2 = c2
#
#     def forward(self, x):
#         if self.conv is not None:
#             x = self.conv(x)
#         b, _, w, h = x.shape
#         p = x.flatten(2).unsqueeze(0).transpose(0, 3).squeeze(3)
#         return self.tr(p + self.linear(p)).unsqueeze(3).transpose(0, 3).reshape(b, self.c2, w, h)


class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, in_channels, out_channels, num_heads, num_layers):
        super(TransformerBlock, self).__init__()
        self.conv = None
        if in_channels != out_channels:
            self.conv = Conv(out_channels=out_channels)
        self.linear = tf.keras.layers.Dense(out_channels)
        self.transformers = tf.keras.Sequential([
            TransformerLayer(c=out_channels, num_heads=num_heads) for _ in range(num_layers)
        ])
        self.out_channels = out_channels

    def call(self, inputs, *args, **kwargs):
        if self.conv is not None:
            """这里如果input的channel不等于参数out_channels, 就通过卷积处理成一样, 方便下面做全连接"""
            inputs = self.conv(inputs)
        in_shape = tf.shape(inputs)
        # [batch, hxw, channels]
        in_flatten = tf.keras.layers.Reshape([in_shape[1] * in_shape[2], in_shape[3]])(inputs)
        # [batch, hxw, channels] -> [1, batch, hxw, channels] -> [hxw, batch, 1, channels] ->
        # [hxw, batch, channels]
        in_transpose = tf.squeeze(tf.transpose(tf.expand_dims(in_flatten, 0), [2, 1, 0, 3]), axis=2)
        out_transpose = self.transformers(in_transpose + self.linear(in_transpose))
        # [hxw, batch, channels] -> [hxw, batch, 1, channels] -> [1, batch, hxw, channels] ->
        # [batch, hxw, channels] -> [batch, h, w, channels]
        out_flatten = tf.squeeze(tf.transpose(tf.expand_dims(out_transpose, axis=2), [2, 1, 0, 3]), axis=0)
        out = tf.keras.layers.Reshape([in_shape[1], in_shape[2], self.out_channels])(out_flatten)
        return out


# class Bottleneck(nn.Module):
#     # Standard bottleneck
#     def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
#         super().__init__()
#         c_ = int(c2 * e)  # hidden channels
#         self.cv1 = Conv(c1, c_, 1, 1)
#         self.cv2 = Conv(c_, c2, 3, 1, g=g)
#         self.add = shortcut and c1 == c2
#
#     def forward(self, x):
#         return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class Bottleneck(tf.keras.layers.Layer):
    def __init__(self, out_channels, shortcut=True, groups=1, expansion=0.5):
        super(Bottleneck, self).__init__()
        self.out_channels = out_channels
        self.shortcut = shortcut
        self.conv1 = Conv(out_channels=int(out_channels * expansion))
        self.conv2 = Conv(out_channels=out_channels, kernel_size=3, stride=1, groups=groups)

    def call(self, inputs, *args, **kwargs):
        # in_shape = tf.shape(inputs)
        in_shape = inputs.get_shape()
        if self.shortcut and in_shape[-1] == self.out_channels:
            return inputs + self.conv2(self.conv1(inputs))
        else:
            return self.conv2(self.conv1(inputs))


# class BottleneckCSP(nn.Module):
#     # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
#     def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
#         super().__init__()
#         c_ = int(c2 * e)  # hidden channels
#         self.cv1 = Conv(c1, c_, 1, 1)
#         self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
#         self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
#         self.cv4 = Conv(2 * c_, c2, 1, 1)
#         self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
#         self.act = nn.LeakyReLU(0.1, inplace=True)
#         self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])
#
#     def forward(self, x):
#         y1 = self.cv3(self.m(self.cv1(x)))
#         y2 = self.cv2(x)
#         return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))


class BottleneckCSP(tf.keras.layers.Layer):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, out_channels, num_bottles=1, shortcut=True, groups=1, expansion=0.5):
        super(BottleneckCSP, self).__init__()
        out_expansion_channels = int(out_channels * expansion)
        self.conv1 = Conv(out_channels=out_expansion_channels)
        self.conv2 = tf.keras.layers.Conv2D(filters=out_expansion_channels, kernel_size=1, use_bias=False)
        self.conv3 = tf.keras.layers.Conv2D(filters=out_expansion_channels, kernel_size=1, use_bias=False)
        self.conv4 = Conv(out_channels=out_channels)
        self.bn = tf.keras.layers.BatchNormalization()
        self.leakRelu = tf.keras.layers.LeakyReLU(alpha=0.1)
        self.bottlenecks = tf.keras.Sequential([
            Bottleneck(out_channels=out_expansion_channels, shortcut=shortcut, groups=groups, expansion=1.0)
            for _ in range(num_bottles)
        ])

    def call(self, inputs, *args, **kwargs):
        y1 = self.conv3(self.bottlenecks(self.conv1(inputs)))
        y2 = self.conv2(inputs)
        y = tf.keras.layers.Concatenate(axis=-1)([y1, y2])
        out = self.conv4(self.leakRelu(self.bn(y)))
        return out


# class C3(nn.Module):
#     # CSP Bottleneck with 3 convolutions
#     def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
#         super().__init__()
#         c_ = int(c2 * e)  # hidden channels
#         self.cv1 = Conv(c1, c_, 1, 1)
#         self.cv2 = Conv(c1, c_, 1, 1)
#         self.cv3 = Conv(2 * c_, c2, 1)  # act=FReLU(c2)
#         self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])
#         # self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])
#
#     def forward(self, x):
#         return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))

class C3(tf.keras.layers.Layer):
    def __init__(self, out_channels, num_bottles=1, shortcut=True, groups=1, expansion=0.5):
        super(C3, self).__init__()
        out_expansion_channels = int(out_channels * expansion)
        self.conv1 = Conv(out_channels=out_expansion_channels)
        self.conv2 = Conv(out_channels=out_expansion_channels)
        self.conv3 = Conv(out_channels=out_channels)
        self.bottlenecks = tf.keras.Sequential([
            Bottleneck(out_channels=out_expansion_channels, shortcut=shortcut, groups=groups, expansion=1.0)
            for _ in range(num_bottles)
        ])

    def call(self, inputs, *args, **kwargs):
        y1 = self.bottlenecks(self.conv1(inputs))
        y2 = self.conv2(inputs)
        y = tf.keras.layers.Concatenate(axis=-1)([y1, y2])
        output = self.conv3(y)
        return output


# class C3TR(C3):
#     # C3 module with TransformerBlock()
#     def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
#         super().__init__(c1, c2, n, shortcut, g, e)
#         c_ = int(c2 * e)
#         self.m = TransformerBlock(c_, c_, 4, n)


class C3TR(C3):
    def __init__(self, in_channels, out_channels, num_bottles=1, shortcut=True, groups=1, expansion=0.5):
        super(C3TR, self).__init__(out_channels, num_bottles, shortcut, groups, expansion)
        out_expansion_channels = int(out_channels * expansion)
        # 这里重写self.bottlenecks方法
        self.bottlenecks = TransformerBlock(
            in_channels=in_channels, out_channels=out_expansion_channels, num_heads=4, num_layers=num_bottles)


# class C3SPP(C3):
#     # C3 module with SPP()
#     def __init__(self, c1, c2, k=(5, 9, 13), n=1, shortcut=True, g=1, e=0.5):
#         super().__init__(c1, c2, n, shortcut, g, e)
#         c_ = int(c2 * e)
#         self.m = SPP(c_, c_, k)


class C3SPP(C3):
    def __init__(self, out_channels, kernels_size=(5, 9, 13), num_bottles=1, shortcut=True, groups=1, expansion=0.5):
        super(C3SPP, self).__init__(out_channels, num_bottles, shortcut, groups, expansion)
        out_expansion_channels = int(out_channels * expansion)
        self.bottlenecks = SPP(in_channels=out_expansion_channels,
                               out_channels=out_expansion_channels,
                               pool_size=kernels_size)


# class C3Ghost(C3):
#     # C3 module with GhostBottleneck()
#     def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
#         super().__init__(c1, c2, n, shortcut, g, e)
#         c_ = int(c2 * e)  # hidden channels
#         self.m = nn.Sequential(*[GhostBottleneck(c_, c_) for _ in range(n)])


class C3Ghost(C3):
    def __init__(self, out_channels, num_bottles=1, shortcut=True, groups=1, expansion=0.5):
        super(C3Ghost, self).__init__(out_channels, num_bottles, shortcut, groups, expansion)
        out_expansion_channels = int(out_channels * expansion)
        self.bottlenecks = tf.keras.Sequential([
            GhostBottleneck(in_channels=out_expansion_channels, out_channels=out_expansion_channels)
            for _ in range(num_bottles)
        ])


# class SPP(nn.Module):
#     # Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729
#     def __init__(self, c1, c2, k=(5, 9, 13)):
#         super().__init__()
#         c_ = c1 // 2  # hidden channels
#         self.cv1 = Conv(c1, c_, 1, 1)
#         self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
#         self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])
#
#     def forward(self, x):
#         x = self.cv1(x)
#         with warnings.catch_warnings():
#             warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
#             return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPP(tf.keras.layers.Layer):
    def __init__(self, in_channels, out_channels, pool_size=(5, 9, 13)):
        super(SPP, self).__init__()
        out_half_in_channels = in_channels // 2
        self.conv1 = Conv(out_channels=out_half_in_channels)
        self.conv2 = Conv(out_channels=out_channels)
        self.maxpools = [tf.keras.layers.MaxPooling2D(pool_size=k, strides=1, padding='same') for k in pool_size]

    def call(self, inputs, *args, **kwargs):
        x = self.conv1(inputs)
        pools = [pool(inputs) for pool in self.maxpools]
        x_concat_pools = tf.keras.layers.Concatenate(axis=-1)([x] + pools)
        out = self.conv2(x_concat_pools)
        return out


# class GhostConv(nn.Module):
#     # Ghost Convolution https://github.com/huawei-noah/ghostnet
#     def __init__(self, c1, c2, k=1, s=1, g=1, act=True):  # ch_in, ch_out, kernel, stride, groups
#         super().__init__()
#         c_ = c2 // 2  # hidden channels
#         self.cv1 = Conv(c1, c_, k, s, None, g, act)
#         self.cv2 = Conv(c_, c_, 5, 1, None, c_, act)
#
#     def forward(self, x):
#         y = self.cv1(x)
#         return torch.cat([y, self.cv2(y)], 1)


class GhostConv(tf.keras.layers.Layer):
    def __init__(self, out_channels, kernel_size=1, stride=1, groups=1, act=True):
        super(GhostConv, self).__init__()
        out_half_channels = out_channels // 2
        self.conv1 = Conv(
            out_channels=out_half_channels, kernel_size=kernel_size, stride=stride, groups=groups, act=act)
        self.conv2 = Conv(
            out_channels=out_half_channels, kernel_size=5, stride=1, groups=out_half_channels, act=act)

    def call(self, inputs, *args, **kwargs):
        y1 = self.conv1(inputs)
        y2 = self.conv2(y1)
        out = tf.keras.layers.Concatenate(axis=-1)([y1, y2])
        return out


# class GhostBottleneck(nn.Module):
#     # Ghost Bottleneck https://github.com/huawei-noah/ghostnet
#     def __init__(self, c1, c2, k=3, s=1):  # ch_in, ch_out, kernel, stride
#         super().__init__()
#         c_ = c2 // 2
#         self.conv = nn.Sequential(GhostConv(c1, c_, 1, 1),  # pw
#                                   DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
#                                   GhostConv(c_, c2, 1, 1, act=False))  # pw-linear
#         self.shortcut = nn.Sequential(DWConv(c1, c1, k, s, act=False),
#                                       Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()
#
#     def forward(self, x):
#         return self.conv(x) + self.shortcut(x)

class GhostBottleneck(tf.keras.layers.Layer):
    # Ghost Bottleneck https://github.com/huawei-noah/ghostnet
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(GhostBottleneck, self).__init__()
        out_half_channels = out_channels // 2
        self.conv = tf.keras.Sequential([
            GhostConv(out_channels=out_half_channels),
            DWConv(in_channels=in_channels, out_channels=out_half_channels, kernel_size=kernel_size, stride=stride)
            if stride == 2 else tf.keras.layers.Lambda(lambda x: x),
            GhostConv(out_channels=out_channels, act=False)
        ])
        self.shortcut = tf.keras.Sequential([
            GhostConv(out_channels=in_channels, kernel_size=kernel_size, stride=stride, act=False),
            Conv(out_channels=out_channels, act=False) if stride == 2 else tf.keras.layers.Lambda(lambda x: x)
        ])

    def call(self, inputs, *args, **kwargs):
        return self.conv(inputs) + self.shortcut(inputs)


# class SPPF(nn.Module):
#     # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
#     def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
#         super().__init__()
#         c_ = c1 // 2  # hidden channels
#         self.cv1 = Conv(c1, c_, 1, 1)
#         self.cv2 = Conv(c_ * 4, c2, 1, 1)
#         self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
#
#     def forward(self, x):
#         x = self.cv1(x)
#         with warnings.catch_warnings():
#             warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
#             y1 = self.m(x)
#             y2 = self.m(y1)
#             return self.cv2(torch.cat([x, y1, y2, self.m(y2)], 1))


class SPPF(tf.keras.layers.Layer):
    def __init__(self, in_channels, out_channels, kernel_size=5):
        super(SPPF, self).__init__()
        in_half_channels = in_channels // 2
        self.conv1 = Conv(in_half_channels)
        self.conv2 = Conv(out_channels)
        self.maxpool = tf.keras.layers.MaxPooling2D(pool_size=(kernel_size, kernel_size), strides=1, padding='same')

    def call(self, inputs, *args, **kwargs):
        x = self.conv1(inputs)
        y1 = self.maxpool(x)
        y2 = self.maxpool(y1)
        y3 = self.maxpool(y2)
        concat_all = tf.keras.layers.Concatenate()([x, y1, y2, y3])
        output = self.conv2(concat_all)
        return output


# class Focus(nn.Module):
#     # Focus wh information into c-space
#     def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
#         super().__init__()
#         self.conv = Conv(c1 * 4, c2, k, s, p, g, act)
#         # self.contract = Contract(gain=2)
#
#     def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
#         return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))
#         # return self.conv(self.contract(x))


class Focus(tf.keras.layers.Layer):
    def __init__(self, out_channels, kernel_size=1, stride=1, padding=None, groups=1, act=True):
        super(Focus, self).__init__()
        self.conv = Conv(out_channels, kernel_size, stride, padding, groups, act)

    def call(self, inputs, *args, **kwargs):
        # [batch, h, w, c] => [batch, h/2, w/2, c]
        half_wh_concat = tf.keras.layers.Concatenate()(
            [inputs[:, ::2, ::2, :], inputs[:, 1::2, ::2, :], inputs[:, ::2, 1::2, :], inputs[:, 1::2, 1::2, :]]
        )
        output = self.conv(half_wh_concat)
        return output


#
# class Contract(nn.Module):
#     # Contract width-height into channels, i.e. x(1,64,80,80) to x(1,256,40,40)
#     def __init__(self, gain=2):
#         super().__init__()
#         self.gain = gain
#
#     def forward(self, x):
#         b, c, h, w = x.size()  # assert (h / s == 0) and (W / s == 0), 'Indivisible gain'
#         s = self.gain
#         x = x.view(b, c, h // s, s, w // s, s)  # x(1,64,40,2,40,2)
#         x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # x(1,2,2,64,40,40)
#         return x.view(b, c * s * s, h // s, w // s)  # x(1,256,40,40)
#

class Contract(tf.keras.layers.Layer):
    def __init__(self, gain=2):
        super(Contract, self).__init__()
        self.gain = gain

    def call(self, inputs, *args, **kwargs):
        in_shape = tf.shape(inputs)
        # [batch, h, w, c] => [batch, h//g, g, w//g, g, c]
        x_reshape = tf.keras.layers.Reshape(
            [in_shape[1] // self.gain, self.gain, in_shape[2] // self.gain, self.gain, in_shape[3]]
        )(inputs)
        # [batch, h//g, g, w//g, g, c] => [batch, h//g, w//g, g, g, c]
        x_transpose = tf.transpose(x_reshape, [0, 1, 3, 2, 4, 5])
        # [batch, h//g, w//g, g, g, c] => [batch, h//g, w//g, g * g * c]
        output = tf.keras.layers.Reshape(
            [in_shape[1] // self.gain, in_shape[2] // self.gain, self.gain * self.gain * in_shape[3]]
        )(x_transpose)
        return output


# class Expand(nn.Module):
#     # Expand channels into width-height, i.e. x(1,64,80,80) to x(1,16,160,160)
#     def __init__(self, gain=2):
#         super().__init__()
#         self.gain = gain
#
#     def forward(self, x):
#         b, c, h, w = x.size()  # assert C / s ** 2 == 0, 'Indivisible gain'
#         s = self.gain
#         x = x.view(b, s, s, c // s ** 2, h, w)  # x(1,2,2,16,80,80)
#         x = x.permute(0, 3, 4, 1, 5, 2).contiguous()  # x(1,16,80,2,80,2)
#         return x.view(b, c // s ** 2, h * s, w * s)  # x(1,16,160,160)

class Expand(tf.keras.layers.Layer):
    def __init__(self, gain):
        super(Expand, self).__init__()
        self.gain = gain

    def call(self, inputs, *args, **kwargs):
        in_shape = tf.shape(inputs)
        # [batch, h, w, c] => [batch, h, w, g, g, c // g**2]
        x_reshape = tf.keras.layers.Reshape(
            [in_shape[1], in_shape[2], self.gain, self.gain, in_shape[3] // self.gain ** 2]
        )(inputs)
        # [batch, h, w, g, g, c // g**2] => [batch, h, g, w, g, c // g**2]
        x_transpose = tf.transpose(x_reshape, [0, 1, 3, 2, 4, 5])
        # [batch, h, g, w, g, c // g**2] => [batch, h * g, w * g, c // g**2]
        output = tf.keras.layers.Reshape(
            [in_shape[1] * self.gain, in_shape[2] * self.gain, in_shape[3] // self.gain ** 2]
        )(x_transpose)
        return output


# class Concat(nn.Module):
#     # Concatenate a list of tensors along dimension
#     def __init__(self, dimension=1):
#         super().__init__()
#         self.d = dimension
#
#     def forward(self, x):
#         return torch.cat(x, self.d)


class Concat(tf.keras.layers.Layer):
    def __init__(self, dimension=3):
        super(Concat, self).__init__()
        self.dimension = dimension

    def call(self, inputs, *args, **kwargs):
        return tf.keras.layers.Concatenate(axis=self.dimension)(inputs)

# class AutoShape(nn.Module):
#     # YOLOv5 input-robust model wrapper for passing cv2/np/PIL/torch inputs. Includes preprocessing, inference and NMS
#     conf = 0.25  # NMS confidence threshold
#     iou = 0.45  # NMS IoU threshold
#     classes = None  # (optional list) filter by class
#     multi_label = False  # NMS multiple labels per box
#     max_det = 1000  # maximum number of detections per image
#
#     def __init__(self, model):
#         super().__init__()
#         self.model = model.eval()
#
#     def autoshape(self):
#         LOGGER.info('AutoShape already enabled, skipping... ')  # model already converted to model.autoshape()
#         return self
#
#     def _apply(self, fn):
#         # Apply to(), cpu(), cuda(), half() to model tensors that are not parameters or registered buffers
#         self = super()._apply(fn)
#         m = self.model.model[-1]  # Detect()
#         m.stride = fn(m.stride)
#         m.grid = list(map(fn, m.grid))
#         if isinstance(m.anchor_grid, list):
#             m.anchor_grid = list(map(fn, m.anchor_grid))
#         return self
#
#     @torch.no_grad()
#     def forward(self, imgs, size=640, augment=False, profile=False):
#         # Inference from various sources. For height=640, width=1280, RGB images example inputs are:
#         #   file:       imgs = 'data/images/zidane.jpg'  # str or PosixPath
#         #   URI:             = 'https://ultralytics.com/images/zidane.jpg'
#         #   OpenCV:          = cv2.imread('image.jpg')[:,:,::-1]  # HWC BGR to RGB x(640,1280,3)
#         #   PIL:             = Image.open('image.jpg') or ImageGrab.grab()  # HWC x(640,1280,3)
#         #   numpy:           = np.zeros((640,1280,3))  # HWC
#         #   torch:           = torch.zeros(16,3,320,640)  # BCHW (scaled to size=640, 0-1 values)
#         #   multiple:        = [Image.open('image1.jpg'), Image.open('image2.jpg'), ...]  # list of images
#
#         t = [time_sync()]
#         p = next(self.model.parameters())  # for device and type
#         if isinstance(imgs, torch.Tensor):  # torch
#             with amp.autocast(enabled=p.device.type != 'cpu'):
#                 return self.model(imgs.to(p.device).type_as(p), augment, profile)  # inference
#
#         # Pre-process
#         n, imgs = (len(imgs), imgs) if isinstance(imgs, list) else (1, [imgs])  # number of images, list of images
#         shape0, shape1, files = [], [], []  # image and inference shapes, filenames
#         for i, im in enumerate(imgs):
#             f = f'image{i}'  # filename
#             if isinstance(im, (str, Path)):  # filename or uri
#                 im, f = Image.open(requests.get(im, stream=True).raw if str(im).startswith('http') else im), im
#                 im = np.asarray(exif_transpose(im))
#             elif isinstance(im, Image.Image):  # PIL Image
#                 im, f = np.asarray(exif_transpose(im)), getattr(im, 'filename', f) or f
#             files.append(Path(f).with_suffix('.jpg').name)
#             if im.shape[0] < 5:  # image in CHW
#                 im = im.transpose((1, 2, 0))  # reverse dataloader .transpose(2, 0, 1)
#             im = im[..., :3] if im.ndim == 3 else np.tile(im[..., None], 3)  # enforce 3ch input
#             s = im.shape[:2]  # HWC
#             shape0.append(s)  # image shape
#             g = (size / max(s))  # gain
#             shape1.append([y * g for y in s])
#             imgs[i] = im if im.data.contiguous else np.ascontiguousarray(im)  # update
#         shape1 = [make_divisible(x, int(self.stride.max())) for x in np.stack(shape1, 0).max(0)]  # inference shape
#         x = [letterbox(im, new_shape=shape1, auto=False)[0] for im in imgs]  # pad
#         x = np.stack(x, 0) if n > 1 else x[0][None]  # stack
#         x = np.ascontiguousarray(x.transpose((0, 3, 1, 2)))  # BHWC to BCHW
#         x = torch.from_numpy(x).to(p.device).type_as(p) / 255.  # uint8 to fp16/32
#         t.append(time_sync())
#
#         with amp.autocast(enabled=p.device.type != 'cpu'):
#             # Inference
#             y = self.model(x, augment, profile)[0]  # forward
#             t.append(time_sync())
#
#             # Post-process
#             y = non_max_suppression(y, self.conf, iou_thres=self.iou, classes=self.classes,
#                                     multi_label=self.multi_label, max_det=self.max_det)  # NMS
#             for i in range(n):
#                 scale_coords(shape1, y[i][:, :4], shape0[i])
#
#             t.append(time_sync())
#             return Detections(imgs, y, files, t, self.names, x.shape)
#
#
# class AutoShape(tf.keras.layers.Layer):
#     def __init__(self):
#         super(AutoShape, self).__init__()
#
#     def call(self, inputs, *args, **kwargs):
#         pass
