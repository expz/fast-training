"""
These classes define the Pervasive Attention 2D deep neural network from
https://arxiv.org/abs/1808.03867.

It is based on the DenseNet convolutional network more commonly used for
image recognition problems.
"""

import math
import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp


class Aggregator(nn.Module):
    """
    This layer takes a max along the source word axis to produce, for each
    output word, a single output per channel.
    """

    def forward(self, x):
        return x.max(dim=3)[0].permute(0, 2, 1)


class MaskedConv2d(nn.Conv2d):
    """
    This is a masked 2D convolutional layer, i.e. A convolution with a filter
    that has been zeroed out to the right of the center axis. This
    prevents information from output tokens after a given output token
    affecting its choice.

    Based on https://github.com/elbayadm/attn2d/blob/master/nmt/models/conv2d.py
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 dilation=1,
                 groups=1,
                 bias=False):
        pad = (dilation * (kernel_size - 1)) // 2
        super(MaskedConv2d, self).__init__(in_channels,
                                           out_channels,
                                           kernel_size,
                                           padding=pad,
                                           groups=groups,
                                           dilation=dilation,
                                           bias=bias)
        _, _, kH, kW = self.weight.size()
        # Use `register_buffer()` so `mask` will be moved to a cuda device with
        # rest of the module.
        self.register_buffer(
            'mask', torch.zeros(self.weight.data.shape, dtype=torch.uint8))
        if kH > 1:
            self.mask[:, :, kH // 2 + 1:, :] = 1
        self.register_buffer(
            'zeros', torch.zeros(self.mask.shape, dtype=self.weight.data.dtype))

    def forward(self, x, *args):
        self.weight.data.masked_scatter_(self.mask, self.zeros)
        return super(MaskedConv2d, self).forward(x)


class DenseLayer(nn.Module):
    """
    Layer of a DenseNet.
    
    Adapted from https://github.com/elbayadm/attn2d/blob/master/nmt/models/efficient_densenet.py
    """

    def __init__(self,
                 input_size,
                 growth_rate,
                 kernel_size=3,
                 bn_size=4,
                 dropout=0,
                 bias=False,
                 efficient=False):
        super(DenseLayer, self).__init__()
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.efficient = efficient

        self.add_module('norm1', nn.BatchNorm2d(input_size)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module(
            'conv1',
            nn.Conv2d(input_size,
                      bn_size * growth_rate,
                      kernel_size=1,
                      bias=False))
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module(
            'conv2',
            MaskedConv2d(bn_size * growth_rate,
                         growth_rate,
                         kernel_size=kernel_size,
                         bias=bias))

    def forward(self, *prev_features):
        """
        Forward pass through network.
        """

        def bottleneck(*inputs):
            """
            Apply the bottleneck sub-layer that collapses to 4 * growth_factor channels.
            Defined so that we can apply `torch.utils.checkpoint.checkpoint`.
            """
            cat_input = torch.cat(inputs, 1)
            return self.conv1(self.relu1(self.norm1(cat_input)))

        calc_grad = any(t.requires_grad
                        for t in prev_features
                        if isinstance(t, torch.Tensor))
        if self.efficient and calc_grad:
            # Wins decreased memory at cost of extra computation. Does not
            # compute intermediate values, but recomputes them in backward pass.
            # Do not checkpoint when running without tracking gradients, e.g. on
            # validation set.
            bottleneck_output = cp.checkpoint(bottleneck, *prev_features)
        else:
            bottleneck_output = bottleneck(*prev_features)
        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.dropout > 0:
            new_features = F.dropout(new_features,
                                     p=self.dropout,
                                     training=self.training)
        return new_features


class DenseBlock(nn.Module):
    """
    Block of layers in a DenseNet.
    
    Adapted from
    https://github.com/elbayadm/attn2d/
            blob/master/nmt/models/efficient_densenet.py
    """
    KERNEL_SIZE = 3

    def __init__(self,
                 num_layers,
                 input_size,
                 bn_size,
                 growth_rate,
                 dropout,
                 bias,
                 efficient=False):
        super(DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = DenseLayer(input_size + i * growth_rate,
                               growth_rate,
                               self.KERNEL_SIZE,
                               bn_size,
                               dropout,
                               efficient=efficient)
            self.add_module(f'denselayer{i + 1}', layer)

    def forward(self, init_features):
        """
        Forward pass through block of layers.
        """
        features = [init_features]
        for layer in self.children():
            new_features = layer(*features)
            features.append(new_features)
        return torch.cat(features, 1)


class Transition(nn.Sequential):
    """
    Transiton layer between dense blocks to reduce number of channels.
    
    BN > ReLU > Conv(k=1)
    
    From
    https://github.com/elbayadm/attn2d/
            blob/master/nmt/models/efficient_densenet.py
    """

    def __init__(self, input_size, output_size):
        super(Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(input_size))
        self.add_module('relu', nn.ReLU(inplace=True))
        conv = nn.Conv2d(input_size, output_size, kernel_size=1, bias=False)
        self.add_module('conv', conv)

    def forward(self, x, *args):
        """Forward pass through layer."""
        return super(Transition, self).forward(x)


class DenseNet(nn.Module):
    """ 
    A `DenseNet` is made of a sequence of one or more `DenseBlock`s each followed
    by a `Transition` convolutional layer.

    The setting `efficient` makes the model much more memory efficient, but slower.
    
    Adapted from
    https://github.com/elbayadm/attn2d/
            blob/master/nmt/models/efficient_densenet.py
    """

    def __init__(self,
                 input_size,
                 block_sizes,
                 bn_size=4,
                 dropout=0.2,
                 growth_rate=32,
                 division_factor=2,
                 bias=False,
                 efficient=False):
        super(DenseNet, self).__init__()

        self.efficient = efficient

        self.model = nn.Sequential()
        num_features = input_size
        if division_factor > 1:
            trans = nn.Conv2d(num_features, num_features // division_factor, 1)
            torch.nn.init.xavier_normal_(trans.weight)
            self.model.add_module('initial_transition', trans)
            num_features = num_features // division_factor

        for i, num_layers in enumerate(block_sizes):
            block = DenseBlock(num_layers=num_layers,
                               input_size=num_features,
                               bn_size=bn_size,
                               growth_rate=growth_rate,
                               dropout=dropout,
                               bias=bias,
                               efficient=efficient)
            self.model.add_module(f'denseblock{i + 1}', block)
            num_features = num_features + num_layers * growth_rate
            trans = Transition(input_size=num_features,
                               output_size=num_features // 2)
            self.model.add_module(f'transition{i + 1}', trans)
            num_features = num_features // 2

        self.output_channels = num_features
        self.model.add_module('norm_final', nn.BatchNorm2d(num_features))
        self.model.add_module('relu_last', nn.ReLU(inplace=True))

    def forward(self, x):
        """Forward pass through network"""
        return self.model(x.contiguous())


class Pervasive(nn.Module):
    """
    Pervasive Attention Network.
    
    Based on
    https://github.com/elbayadm/attn2d/
            blob/master/nmt/models/pervasive.py
    """

    PAD = 0

    def __init__(self,
                 name,
                 src_vocab,
                 tgt_vocab,
                 block_sizes,
                 Ts=50,
                 Tt=50,
                 src_emb_size=128,
                 tgt_emb_size=128,
                 enc_dropout=0.2,
                 conv_dropout=0.2,
                 dec_dropout=0.2,
                 prediction_dropout=0.2,
                 division_factor=2,
                 growth_rate=32,
                 bias=False,
                 efficient=False):
        nn.Module.__init__(self)
        self.src_vocab_size = src_vocab.vocab_size
        self.tgt_vocab_size = tgt_vocab.vocab_size
        self.Ts = Ts
        self.Tt = Tt
        self.enc_dropout = enc_dropout
        self.dec_dropout = dec_dropout
        self.src_embedding = nn.Embedding(self.src_vocab_size, src_emb_size)
        self.tgt_embedding = nn.Embedding(self.tgt_vocab_size, tgt_emb_size)

        self.input_channels = src_emb_size + tgt_emb_size

        self.densenet = DenseNet(self.input_channels,
                                 block_sizes,
                                 dropout=conv_dropout,
                                 growth_rate=growth_rate,
                                 division_factor=division_factor,
                                 bias=bias,
                                 efficient=efficient)

        self.aggregator = Aggregator()

        self.linear = nn.Linear(self.densenet.output_channels, tgt_emb_size)
        self.relu = nn.ReLU(inplace=True)

        self.prediction_dropout = nn.Dropout(prediction_dropout)
        self.prediction = nn.Linear(tgt_emb_size, self.tgt_vocab_size)
        self.prediction.weight = self.tgt_embedding.weight

    def init_weights(self):
        """
        Initialize weights of all submodules.
        """
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                # Tensorflow default embedding initialization
                # (except they resample instead of clipping).
                input_size = m.weight.shape[1]
                std = 1 / math.sqrt(input_size)
                nn.init.normal_(m.weight, 0, std)
                torch.clamp_(m.weight.data, min=-2 * std, max=2 * std)
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)

            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, data):
        """
        Forward pass of `data` through the network.
        """
        src_data, tgt_data = data.split([self.Ts, self.Tt], dim=1)
        src_emb = F.dropout(self.src_embedding(src_data), p=self.enc_dropout)
        tgt_emb = F.dropout(self.tgt_embedding(tgt_data), p=self.dec_dropout)
        src_emb = src_emb.unsqueeze(1).repeat(1, self.Tt, 1, 1)
        tgt_emb = tgt_emb.unsqueeze(2).repeat(1, 1, self.Ts, 1)

        X = torch.cat((src_emb, tgt_emb), dim=3)
        X = X.permute(0, 3, 1, 2)
        X = self.densenet(X)
        X = self.aggregator(X)
        X = self.relu(self.linear(X))
        X = self.prediction(self.prediction_dropout(X))

        logits = F.log_softmax(X, dim=2)
        return logits.permute(0, 2, 1)
