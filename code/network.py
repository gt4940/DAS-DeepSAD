import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepSAD(nn.Module):
    def __init__(self, args):
        super(DeepSAD, self).__init__()
        self.num_filter = args.num_filter
        self.latent_dim = args.latent_dim
        act_fn = nn.LeakyReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.down_1 = conv_block(1, self.num_filter, act_fn)
        self.down_2 = conv_block(self.num_filter * 1, self.num_filter * 2, act_fn)
        self.down_3 = conv_block(self.num_filter * 2, self.num_filter * 4, act_fn)
        self.down_4 = conv_block(self.num_filter * 4, self.num_filter * 8, act_fn)

        self.bridge = conv_block_no_bn(self.num_filter * 8, self.num_filter * 16, act_fn)
        self.flatten = nn.Flatten()
        self.linear_1 = nn.Linear(self.num_filter * 16 * 62 * 25, self.latent_dim, bias=False)

    def forward(self, x):
        down_1 = self.down_1(x)
        pool_1 = self.pool(down_1)
        down_2 = self.down_2(pool_1)
        pool_2 = self.pool(down_2)
        down_3 = self.down_3(pool_2)
        pool_3 = self.pool(down_3)
        down_4 = self.down_4(pool_3)
        pool_4 = self.pool(down_4)

        bridge = self.bridge(pool_4)
        flatten = self.flatten(bridge)
        bottleneck = self.linear_1(flatten)
        return bottleneck


class UNet(nn.Module):
    def __init__(self, args):
        super(UNet, self).__init__()
        self.num_filter = args.num_filter
        self.latent_dim = args.latent_dim
        act_fn = nn.LeakyReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        print("\n------Initiating U-Net------\n")

        self.down_1 = conv_block(1, self.num_filter, act_fn)
        self.down_2 = conv_block(self.num_filter * 1, self.num_filter * 2, act_fn)
        self.down_3 = conv_block(self.num_filter * 2, self.num_filter * 4, act_fn)
        self.down_4 = conv_block(self.num_filter * 4, self.num_filter * 8, act_fn)
        self.bridge = conv_block_no_bn(self.num_filter * 8, self.num_filter * 16, act_fn)

        self.flatten = nn.Flatten()
        self.linear_1 = nn.Linear(self.num_filter * 16 * 62 * 25, self.latent_dim, bias=False)
        self.linear_2 = nn.Linear(self.latent_dim, self.num_filter * 16 * 62 * 25, bias=False)

        self.trans_1 = conv_trans_block(self.num_filter * 16, self.num_filter * 8, act_fn)
        self.trans_2 = conv_trans_block(self.num_filter * 8, self.num_filter * 4, act_fn)
        self.trans_3 = conv_trans_block(self.num_filter * 4, self.num_filter * 2, act_fn)
        self.trans_4 = conv_trans_block(self.num_filter * 2, self.num_filter * 1, act_fn)

        self.up_1 = conv_block_no_bn(self.num_filter * 16, self.num_filter * 8, act_fn)
        self.up_2 = conv_block_no_bn(self.num_filter * 8, self.num_filter * 4, act_fn)
        self.up_3 = conv_block_no_bn(self.num_filter * 4, self.num_filter * 2, act_fn)
        self.up_4 = conv_block_no_bn(self.num_filter * 2, self.num_filter * 1, act_fn)

        self.out = nn.Conv2d(self.num_filter, 1, 1, bias=False)

    def encoder(self, input):
        down_1 = self.down_1(input)
        pool_1 = self.pool(down_1)
        down_2 = self.down_2(pool_1)
        pool_2 = self.pool(down_2)
        down_3 = self.down_3(pool_2)
        pool_3 = self.pool(down_3)
        down_4 = self.down_4(pool_3)
        pool_4 = self.pool(down_4)
        bridge = self.bridge(pool_4)
        flatten = self.flatten(bridge)
        bottleneck = self.linear_1(flatten)
        return bottleneck

    def forward(self, input):
        down_1 = self.down_1(input)
        pool_1 = self.pool(down_1)
        down_2 = self.down_2(pool_1)
        pool_2 = self.pool(down_2)
        down_3 = self.down_3(pool_2)
        pool_3 = self.pool(down_3)
        down_4 = self.down_4(pool_3)
        pool_4 = self.pool(down_4)
        bridge = self.bridge(pool_4)
        flatten = self.flatten(bridge)
        bottleneck = self.linear_1(flatten)
        bottleneck = self.linear_2(bottleneck)
        reshaped = bottleneck.view(bridge.size(0), bridge.size(1), bridge.size(2), bridge.size(3))

        trans_1 = self.trans_1(reshaped)
        concat_1 = cutandcat(trans_1, down_4)
        up_1 = self.up_1(concat_1)
        trans_2 = self.trans_2(up_1)
        concat_2 = cutandcat(trans_2, down_3)
        up_2 = self.up_2(concat_2)
        trans_3 = self.trans_3(up_2)
        concat_3 = cutandcat(trans_3, down_2)
        up_3 = self.up_3(concat_3)
        trans_4 = self.trans_4(up_3)
        concat_4 = cutandcat(trans_4, down_1)
        up_4 = self.up_4(concat_4)
        out = self.out(up_4)
        return out


def cutandcat(de, en):
    diffY = en.size()[2] - de.size()[2]
    diffX = en.size()[3] - de.size()[3]
    de = F.pad(de, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2], 'reflect')
    cat = torch.cat([en, de], dim=1)
    return cat


def conv_block(in_dim, out_dim, act_fn):
    model = nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1, bias=False),
                          nn.BatchNorm2d(out_dim),
                          act_fn,
                          nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1, bias=False),
                          nn.BatchNorm2d(out_dim),
                          act_fn
                          )
    return model


def conv_block_no_bn(in_dim, out_dim, act_fn):
    model = nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1, bias=False),
                          act_fn,
                          nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1, bias=False),
                          act_fn
                          )
    return model


def conv_trans_block(in_dim, out_dim, act_fn):
    model = nn.Sequential(nn.ConvTranspose2d(in_dim, out_dim, kernel_size=2, stride=2, padding=0, bias=False),
                          act_fn
                          )
    return model
