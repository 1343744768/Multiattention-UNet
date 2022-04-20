import torch
import torch.nn as nn
#from torchsummary import summary


class Attention(nn.Module):
    def __init__(self,
                 dim,  # 输入token的dim
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        # [batch_size, num_patches + 1, total_embed_dim]
        B, N, C = x.shape

        # qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim]
        # reshape: -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches + 1, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class simam_module(torch.nn.Module):
    def __init__(self, channels=None, e_lambda=1e-4):
        super(simam_module, self).__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "simam"

    def forward(self, x):
        b, c, h, w = x.size()

        n = w * h - 1

        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5

        return x * self.activaton(y)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


def conv3x3(in_planes, out_planes):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1)

class sam_residual(nn.Module):
    def __init__(self, inplanes, planes):
        super(sam_residual, self).__init__()
        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Sequential(conv3x3(planes, planes), simam_module(planes))
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=1),
            nn.BatchNorm2d(planes),
        )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class unetUp(nn.Module):
    def __init__(self, in_size, mid_size, out_size, upsize, att_type='channel'):  #channel or spatial
        super(unetUp, self).__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(upsize, upsize, 4, 2, 1, bias=False),
            nn.BatchNorm2d(upsize),
            nn.ReLU(True)
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_size, mid_size, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_size),
            nn.ReLU(True)
        )

        if att_type == 'channel':
            self.att = ChannelAttention(mid_size)
        elif att_type == 'spatial':
            self.att = SpatialAttention()
        else:
            self.att = nn.Identity()

        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_size, out_size, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_size),
            nn.ReLU(True)
        )

    def forward(self, inputs1, inputs2):
        outputs = torch.cat([inputs1, self.up(inputs2)], 1)
        outputs = self.conv1(outputs)
        outputs = self.att(outputs) * outputs
        outputs = self.conv2(outputs)
        return outputs


class Unet(nn.Module):
    def __init__(self, num_classes=21):
        super(Unet, self).__init__()
        self.conv0 = sam_residual(3, 64)                                    # 64 512

        self.down1 = nn.Sequential(
                        nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 128 256
                        sam_residual(128, 128))

        self.down2 = nn.Sequential(
                        nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1) , # 128 128
                        sam_residual(128, 128))

        self.down3 = nn.Sequential(
                        nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # 256 64
                        sam_residual(256, 256))

        self.down4 = nn.Sequential(
                        nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),  # 256 32
                        sam_residual(256, 256))

        self.down5_1 = nn.Sequential(
                        nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
                        nn.BatchNorm2d(512),
                        nn.ReLU(True))

        self.down5_2 = nn.Sequential(
                        nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm2d(256),
                        nn.ReLU(True)
        )
        self.att = nn.Sequential(
                        nn.LayerNorm(512),
                        Attention(512)
        )
        # upsampling  #channel or spatial
        self.up_concat4 = unetUp(512, 256, 256, upsize=256, att_type='channel')
        # (256,16,16)-->(256,32,32) cat (256,32,32) = (512,32,32) -->(256,32,32)-->(256,32,32)
        self.up_concat3 = unetUp(512, 256, 128, upsize=256, att_type='channel')
        # (256,32,32)-->(256,64,64) cat (256,64,64) = (512,64,64) -->(256,64,64)-->(128,64,64)
        self.up_concat2 = unetUp(256, 128, 128, upsize=128, att_type='spatial')
        # (128,64,64)-->(128,128,128) cat (128,128,128) = (256,128,128) -->(128,128,128)-->(128,128,128)
        self.up_concat1 = unetUp(256, 128, 64, upsize=128, att_type='spatial')
        # (128,128,128)-->(128,256,256) cat (128,256,256) = (256,256,256) -->(128,256,256)-->(64,256,256)
        self.up_concat0 = unetUp(128, 64, 64, upsize=64, att_type='spatial')
        # (64,256,256)-->(64,512,512) cat (64,512,512) = (128,512,512) -->(64,512,512)-->(64,512,512)

        #self.att = nn.Sequential(

        self.final = nn.Conv2d(64, num_classes, 1)

    def forward(self, inputs):
        h, w = inputs.shape[2], inputs.shape[3]
        feat0 = self.conv0(inputs) # (64, 512, 512)
        feat1 = self.down1(feat0)  # (128, 256, 256)
        feat2 = self.down2(feat1)  # (128, 128, 128)
        feat3 = self.down3(feat2)  # (256, 64, 64)
        feat4 = self.down4(feat3)  # (256, 32, 32)
        feat5 = self.down5_1(feat4)  # (512, 16, 16)
        feat5 = feat5.flatten(2).transpose(1, 2)
        feat5 = feat5 + self.att(feat5)
        feat5 = feat5.transpose(1, 2).view(-1, 512, int(h / 32), int(w / 32))
        feat5 = self.down5_2(feat5)  # (256, 16, 16)
        up4 = self.up_concat4(feat4, feat5)  # (256, 32, 32)
        up3 = self.up_concat3(feat3, up4)
        up2 = self.up_concat2(feat2, up3)
        up1 = self.up_concat1(feat1, up2)
        up0 = self.up_concat0(feat0, up1)
        final = self.final(up0)

        return final

    def _initialize_weights(self, *stages):
        for modules in stages:
            for module in modules.modules():
                if isinstance(module, nn.Conv2d):
                    nn.init.kaiming_normal_(module.weight)
                    if module.bias is not None:
                        module.bias.data.zero_()
                elif isinstance(module, nn.BatchNorm2d):
                    module.weight.data.fill_(1)
                    module.bias.data.zero_()

#model = Unet()
#print(model)
#summary(model, input_size=(3, 512, 512), device='cpu')
# a = torch.randn(1, 3, 800, 800)
# y = model(a)
# print(y.size())

