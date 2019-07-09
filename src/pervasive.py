"""
These classes define the Pervasive Attention 2D deep neural network from
https://arxiv.org/abs/1808.03867.

The plain `Pervasive` model was coded with recourse to the author's example
code at https://github.com/elbayadm/attn2d.

The models are based on the DenseNet convolutional network more commonly used
for image recognition problems.
"""

import functools
import math
from collections import OrderedDict
from pytorch_pretrained_bert import BertModel
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.utils.checkpoint as cp


# Cache most recent function result using @cache decorator.
cache = functools.lru_cache(1)


class Aggregator(nn.Module):
    """
    This layer takes a max along the source word axis to produce, for each
    output word, a single output per channel.
    """

    def forward(self, x):
        return x.max(dim=3)[0].permute(0, 2, 1)


def dilate(network, fill_with_avg=True):
    """
    This function takes in a `PervasiveNetwork` with convolutions of size 3
    and dilates them to size 5. It fills in the gaps of the newly created
    5 x 5 filter by averaging neighboring cells.
    """
    for name, m in network.densenet.named_children():
        if 'denseblock' in name:
            for name2, m2 in m.named_children():
                if 'denselayer' in name2:
                    assert m2.conv2.kernel_size == 3
                    m2.conv2.kernel_size = 5
                    m2.conv2.padding = 2
                    w = torch.Tensor(
                        m2.conv2.in_channels,
                        m2.conv2.out_channels // m2.conv2.groups, 5, 5)
                    nn.init.kaiming_uniform_(w, nonlinearity='relu')
                    for i in range(3):
                        for j in range(3):
                            w[:, :, i * 2, j * 2] = \
                                m2.conv2.weight.data[:, :, i, j]
                            if fill_with_avg:
                                w[:, :, i + 1, j + 1] = \
                                    (m2.conv2.weight.data[:, :, i, j]
                                        + m2.conv2.weight.data[:, :, 1, 1]) / 2
                    if fill_with_avg:
                        for i, j in [(0, 1), (0, 3), (1, 4), (3, 4)]:
                            w[:, :, i, j] = \
                                (m2.conv2.weight.data[:, :, i, j - 1]
                                    + m2.conv2.weight.data[:, :, i, j + 1]) / 2
                        for i, j in [(1, 0), (3, 0),  (4, 3), (4, 1)]:
                            w[:, :, i, j] = \
                                (m2.conv2.weight.data[:, :, i - 1, j]
                                    + m2.conv2.weight.data[:, :, i + 1, j]) / 2
                    m2.conv2.weight = Parameter(w)
                    m2.conv2.k = 5
                    m2.conv2.register_buffer(
                        'mask', torch.zeros((5, 5), dtype=torch.uint8))
                    m2.conv2.mask[:, :, 3:, :] = 1
                    m2.conv2.register_buffer(
                        'zeros', torch.zeros(
                            (5, 5), dtype=m2.conv2.weight.data.dtype))


class MaskedConv(nn.Module):
    """
    This is a base class for 2D masked convolutions.
    """

    def __init__(self, ConvClass, in_channels, out_channels, kernel_size=3,
                 pad=0, stride=1, dilation=1, groups=1, bias=False):

        self.ConvClass = ConvClass
        ConvClass.__init__(
            self, in_channels, out_channels, kernel_size, padding=pad, stride=1,
            groups=groups, dilation=dilation, bias=bias)

        _, _, self.k, _ = self.weight.size()
        # Use `register_buffer()` so `mask` will be moved to a cuda device with
        # rest of the module.
        self.register_buffer(
            'mask', torch.zeros(self.weight.data.shape, dtype=torch.uint8))
        if self.k > 1:
            self.mask[:, :, self.k // 2 + 1:, :] = 1
        self.register_buffer(
            'zeros', torch.zeros(self.mask.shape, dtype=self.weight.data.dtype))

    def forward(self, x, *args):
        self.weight.data.masked_scatter_(self.mask, self.zeros)
        return self.ConvClass.forward(self, x)


class MaskedConv2d(MaskedConv, nn.Conv2d):
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
                 pad=None,
                 stride=1,
                 dilation=1,
                 groups=1,
                 bias=False):
        pad = (dilation * (kernel_size - 1)) // 2 if pad is None else pad
        MaskedConv.__init__(
            self, nn.Conv2d, in_channels, out_channels, kernel_size, pad=pad,
            stride=1, groups=groups, dilation=dilation, bias=bias)


class MaskedConvTranspose2d(MaskedConv, nn.ConvTranspose2d):
    """
    This is a masked 2D transpose convolutional layer, i.e. A transpose
    convolution with a filter that has been zeroed out to the right of the
    center axis. This prevents information from output tokens after a given
    output token affecting its choice.

    Based on https://github.com/elbayadm/attn2d/blob/master/nmt/models/conv2d.py
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 pad=0,
                 stride=1,
                 dilation=1,
                 groups=1,
                 bias=False):
        MaskedConv.__init__(
            self, nn.ConvTranspose2d, in_channels, out_channels, kernel_size,
            pad=pad, stride=1, groups=groups, dilation=dilation, bias=bias)


class DenseLayer(nn.Module):
    """
    Layer of a DenseNet.

    Adapted from https://github.com/elbayadm/attn2d/blob/master/nmt/models/efficient_densenet.py  # noqa: E501
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

        Adapted from
        https://github.com/elbayadm/attn2d/blob/master/nmt/models/pervasive.py
        """

        def bottleneck(*inputs):
            """
            Apply the bottleneck sub-layer that collapses to 4 * growth_factor
            channels. Defined so that we can apply
            `torch.utils.checkpoint.checkpoint`.

            Adapted from https://github.com/elbayadm/attn2d/blob/master/nmt/models/efficient_densenet.py  # noqa: E501
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
    https://github.com/elbayadm/attn2d/blob/master/nmt/models/efficient_densenet.py  # noqa: E501
    """

    def __init__(self,
                 num_layers,
                 input_size,
                 bn_size,
                 growth_rate,
                 dropout,
                 bias,
                 efficient=False,
                 kernel_size=3):
        super(DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = DenseLayer(input_size + i * growth_rate,
                               growth_rate,
                               kernel_size,
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
    https://github.com/elbayadm/attn2d/blob/master/nmt/models/efficient_densenet.py  # noqa: E501
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
    A `DenseNet` is made of a sequence of one or more `DenseBlock`s each
    followed by a `Transition` convolutional layer.

    The setting `efficient` makes the model much more memory efficient, but
    slower.

    Adapted from
    https://github.com/elbayadm/attn2d/blob/master/nmt/models/efficient_densenet.py  # noqa: E501
    """

    def __init__(self,
                 input_size,
                 block_sizes,
                 bn_size=4,
                 dropout=0.2,
                 growth_rate=32,
                 division_factor=2,
                 bias=False,
                 efficient=False,
                 kernel_size=3):
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
                               efficient=efficient,
                               kernel_size=3)
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


class PervasiveNetwork(nn.Sequential):
    """Encapsulates the non-embedding portion of a Pervasive neural network."""

    def __init__(self,
                 block_sizes,
                 Ts=51,
                 Tt=51,
                 emb_size=128,
                 conv_dropout=0.2,
                 division_factor=2,
                 growth_rate=32,
                 bias=False,
                 efficient=False,
                 kernel_size=3,
                 downsample=False):
        self.Ts = Ts
        self.Tt = Tt
        self.input_channels = 2 * emb_size

        layers = OrderedDict([])

        if downsample:
            layers['downsample'] = MaskedConv2d(
                2 * emb_size, 2 * emb_size, kernel_size=3, pad=1, stride=2,
                bias=True)

        layers['densenet'] = DenseNet(self.input_channels,
                                      block_sizes,
                                      dropout=conv_dropout,
                                      growth_rate=growth_rate,
                                      division_factor=division_factor,
                                      bias=bias,
                                      efficient=efficient,
                                      kernel_size=kernel_size)

        # Masked transpose convolution gives access to information
        # about future tokens at pixels falling between input pixels.
        #
        # if downsample:
        #    layers['upsample'] = MaskedConvTranspose2d(
        #            2 * emb_size, 2 * emb_size, kernel_size=3, pad=1, stride=2,
        #            bias=True),
        #    ]))

        layers['aggregator'] = Aggregator()

        layers['linear'] = nn.Linear(
            layers['densenet'].output_channels, emb_size)
        layers['relu'] = nn.ReLU(inplace=True)

        super().__init__(layers)


class PermutationLayer(nn.Module):
    """
    This class allows performing permutations in modules defined using the
    nn.Sequential constructor.
    """

    def __init__(self, *dims):
        nn.Module.__init__(self)
        self.dims = dims

    def forward(self, X):
        return X.permute(*self.dims)


class PervasiveOriginal(nn.Module):
    """
    Pervasive Attention Network.

    This is a reimplementation of the model in
    https://github.com/elbayadm/attn2d/blob/master/nmt/models/pervasive.py  # noqa: E501
    """

    def __init__(self,
                 block_sizes,
                 vocab_sz,
                 bos,
                 Ts=51,
                 Tt=51,
                 emb_size=128,
                 emb_dropout=0.2,
                 conv_dropout=0.2,
                 prediction_dropout=0.2,
                 division_factor=2,
                 growth_rate=32,
                 bias=False,
                 efficient=False,
                 kernel_size=3):
        nn.Module.__init__(self)

        self.bos = bos
        self.Ts = Ts
        self.Tt = Tt

        self.emb_dropout = nn.Dropout(emb_dropout)
        self.embedding = nn.Embedding(vocab_sz, emb_size)

        self.network = PervasiveNetwork(
            block_sizes, Ts, Tt, emb_size, conv_dropout, division_factor,
            growth_rate, bias, efficient, kernel_size)

        self.prediction_dropout = nn.Dropout(prediction_dropout)
        self.prediction = nn.Linear(emb_size, vocab_sz)
        self.prediction.weight = self.embedding.weight

    def init_weights(self):
        """
        Initialize weights of all submodules.
        """
        for m in self.modules():
            if hasattr(m, 'weight') and not m.weight.requires_grad:
                # Skip fixed layers, since they were intialized manually.
                continue
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
        Forward pass of `data` through network in a training context.
        """
        Tt = data.shape[1] - self.Ts
        src_data, tgt_data = data.split([self.Ts, Tt], dim=1)
        X = self._forward(src_data, tgt_data)

        logits = F.log_softmax(X, dim=2)
        return logits.permute(0, 2, 1)

    def _forward(self, src_data, tgt_data):
        """
        Forward pass of `data` through the entire model.
        """
        # Embedding.
        src_emb = self.emb_dropout(self.embedding(src_data))
        tgt_emb = self.emb_dropout(self.embedding(tgt_data))
        src_grid = src_emb.unsqueeze(1).repeat(1, tgt_emb.shape[1], 1, 1)
        tgt_grid = tgt_emb.unsqueeze(2).repeat(1, 1, src_emb.shape[1], 1)
        X = torch.cat((src_grid, tgt_grid), dim=3)

        # Embedding dim becomes channels.
        X = X.permute(0, 3, 1, 2)

        # Apply network and predict.
        X = self.network(X)
        X = self.prediction(self.prediction_dropout(X))
        return X

    @cache
    def get_bos_col(self, batch_size, device):
        """
        This functions allows us to define a column of BOS tokens
        once for all using caching.
        """
        return torch.tensor([[self.bos]], dtype=torch.int64).repeat(
            batch_size, 1).to(device)

    def predict(self, src_data, tgt_data=None):
        """
        Predict next output token for each batch example based on `src_data`
        and the previous output `tgt_data`. The output `tgt_data` should
        not contain beginning of sequence (BOS) tokens.
        """
        tgt_cols = self.get_bos_col(src_data.shape[0], src_data.device)
        if tgt_data is not None:
            tgt_cols = torch.cat((tgt_cols, tgt_data), dim=1)
        X = self._forward(src_data, tgt_cols)
        logits = F.log_softmax(X, dim=2)
        return logits[:, -1, :]


class Pervasive(PervasiveOriginal):
    """
    A modificiation of the pervasive attention model at

    https://github.com/elbayadm/attn2d/blob/master/nmt/models/pervasive.py

    Using the default settings, it takes in embedding vector of size 768
    and a linear layer reduces them to size 128. From there the model is
    the same until the output layer where a linear layer increases the
    output from 128 dimensions back to 768.

    This is the model used by translation scripts, because it allows the
    use of Bert embeddings, but it does not require loading the Bert
    model, because its weights are meant to be loaded from a saved
    `PervasiveBert` model.
    """

    def __init__(self,
                 block_sizes,
                 vocab_sz,
                 bos,
                 Ts=51,
                 Tt=51,
                 initial_emb_size=768,
                 emb_size=128,
                 emb_dropout=0.2,
                 conv_dropout=0.2,
                 prediction_dropout=0.2,
                 division_factor=2,
                 growth_rate=32,
                 bias=False,
                 efficient=False,
                 kernel_size=3):
        nn.Module.__init__(self)

        self.bos = bos
        self.Ts = Ts
        self.Tt = Tt

        self.emb_dropout = nn.Dropout(emb_dropout)
        self.embedding = nn.Embedding(vocab_sz, initial_emb_size)

        # Embed and project to a lower-dimensional embedding.
        self.projection = nn.Sequential(OrderedDict([
            ('projection_dropout', nn.Dropout(emb_dropout)),
            ('projection', nn.Linear(initial_emb_size, emb_size)),
            ('projection_perm1', PermutationLayer(0, 2, 1)),
            ('projection_norm', nn.BatchNorm1d(emb_size)),
            ('projection_perm2', PermutationLayer(0, 2, 1)),
            ('projection_relu', nn.ReLU(inplace=True)),
        ]))

        # Source and target embeddings will be concatenated to form input.
        self.network = PervasiveNetwork(
            block_sizes, Ts, Tt, emb_size, conv_dropout, division_factor,
            growth_rate, bias, efficient, kernel_size)

        # Unprojection layer.
        self.unprojection = nn.Sequential(OrderedDict([
            ('unprojection_dropout', nn.Dropout(emb_dropout)),
            ('unprojection', nn.Linear(emb_size, initial_emb_size)),
            ('unprojection_perm1', PermutationLayer(0, 2, 1)),
            ('unprojection_norm', nn.BatchNorm1d(initial_emb_size)),
            ('unprojection_perm2', PermutationLayer(0, 2, 1)),
            ('unprojection_relu', nn.ReLU(inplace=True)),
        ]))

        # Output layer.
        self.prediction_dropout = nn.Dropout(prediction_dropout)
        self.prediction = \
            nn.Linear(initial_emb_size, vocab_sz)
        self.prediction.weight = self.embedding.weight
        self.prediction.weight.requires_grad = False

    def _forward(self, src_data, tgt_data):
        """
        Forward pass of `data` through the entire model.
        """
        # Embedding with projection.
        src_emb = self.projection(self.embedding(src_data))
        tgt_emb = self.projection(self.embedding(tgt_data))

        # Prepare grid where embedding dim becomes channels.
        src_grid = src_emb.unsqueeze(1).repeat(1, tgt_emb.shape[1], 1, 1)
        tgt_grid = tgt_emb.unsqueeze(2).repeat(1, 1, src_emb.shape[1], 1)
        X = torch.cat((src_grid, tgt_grid), dim=3)
        X = X.permute(0, 3, 1, 2)

        # Pass data through DenseNet.
        X = self.network(X)

        # Unprojection layer.
        X = self.unprojection(X)

        X = self.prediction(self.prediction_dropout(X))
        return X


class PervasiveBert(Pervasive):
    """
    This class is a Pervasive neural network which uses BERT pre-trained
    multilingual, cased embeddings. The embeddings are 768-dimensional
    and are projected down to 128 dimension before being fed into the network.
    """

    def __init__(self,
                 block_sizes,
                 bos,
                 Ts=51,
                 Tt=51,
                 emb_size=128,
                 emb_dropout=0.2,
                 conv_dropout=0.2,
                 prediction_dropout=0.2,
                 division_factor=2,
                 growth_rate=32,
                 bias=False,
                 efficient=False,
                 kernel_size=3):
        nn.Module.__init__(self)

        self.bos = bos
        self.Ts = Ts
        self.Tt = Tt

        # Load the BERT 768-dim embeddings into a frozen layer.
        bert_model = BertModel.from_pretrained('bert-base-multilingual-cased')
        bert_emb = bert_model.embeddings.word_embeddings
        self.embedding = nn.Embedding.from_pretrained(bert_emb.weight)

        # Embed and project to a lower-dimensional embedding.
        self.projection = nn.Sequential(OrderedDict([
            ('projection_dropout', nn.Dropout(emb_dropout)),
            ('projection', nn.Linear(bert_emb.embedding_dim, emb_size)),
            ('projection_perm1', PermutationLayer(0, 2, 1)),
            ('projection_norm', nn.BatchNorm1d(emb_size)),
            ('projection_perm2', PermutationLayer(0, 2, 1)),
            ('projection_relu', nn.ReLU(inplace=True)),
        ]))

        # Source and target embeddings will be concatenated to form input.
        self.network = PervasiveNetwork(
            block_sizes, Ts, Tt, emb_size, conv_dropout, division_factor,
            growth_rate, bias, efficient, kernel_size)

        # Unprojection layer.
        self.unprojection = nn.Sequential(OrderedDict([
            ('unprojection_dropout', nn.Dropout(emb_dropout)),
            ('unprojection', nn.Linear(emb_size, bert_emb.embedding_dim)),
            ('unprojection_perm1', PermutationLayer(0, 2, 1)),
            ('unprojection_norm', nn.BatchNorm1d(bert_emb.embedding_dim)),
            ('unprojection_perm2', PermutationLayer(0, 2, 1)),
            ('unprojection_relu', nn.ReLU(inplace=True)),
        ]))

        # Output layer.
        self.prediction_dropout = nn.Dropout(prediction_dropout)
        self.prediction = \
            nn.Linear(bert_emb.embedding_dim, bert_emb.num_embeddings)
        self.prediction.weight = self.embedding.weight
        self.prediction.weight.requires_grad = False


class PervasiveDownsample(PervasiveBert):
    """
    This class is a Pervasive neural network which uses BERT pre-trained
    multilingual, cased embeddings. The embeddings are 768-dimensional
    and are projected down to 128 dimension before being fed into the network.
    """

    def __init__(self,
                 block_sizes,
                 bos,
                 Ts=51,
                 Tt=51,
                 emb_size=128,
                 emb_dropout=0.2,
                 conv_dropout=0.2,
                 prediction_dropout=0.2,
                 division_factor=2,
                 growth_rate=32,
                 bias=False,
                 efficient=False,
                 kernel_size=3):
        nn.Module.__init__(self)

        self.bos = bos
        self.Ts = Ts
        self.Tt = Tt

        # Load the BERT 768-dim embeddings into a frozen layer.
        bert_model = BertModel.from_pretrained('bert-base-multilingual-cased')
        bert_emb = bert_model.embeddings.word_embeddings
        self.embedding = nn.Embedding.from_pretrained(bert_emb.weight)

        # Embed and project to a lower-dimensional embedding.
        self.projection = nn.Sequential(OrderedDict([
            ('projection_dropout', nn.Dropout(emb_dropout)),
            ('projection', nn.Linear(bert_emb.embedding_dim, emb_size)),
            ('projection_perm1', PermutationLayer(0, 2, 1)),
            ('projection_norm', nn.BatchNorm1d(emb_size)),
            ('projection_perm2', PermutationLayer(0, 2, 1)),
            ('projection_relu', nn.ReLU(inplace=True)),
        ]))

        # Source and target embeddings will be concatenated to form input.
        self.network = PervasiveNetwork(
            block_sizes, Ts // 2 + 1, Tt // 2 + 1, emb_size, conv_dropout,
            division_factor, growth_rate, bias, efficient, kernel_size,
            downsample=True)

        # Unprojection layer.
        self.unprojection = nn.Sequential(OrderedDict([
            ('unprojection_dropout', nn.Dropout(emb_dropout)),
            ('unprojection', nn.Linear(emb_size, bert_emb.embedding_dim)),
            ('unprojection_perm1', PermutationLayer(0, 2, 1)),
            ('unprojection_norm', nn.BatchNorm1d(bert_emb.embedding_dim)),
            ('unprojection_perm2', PermutationLayer(0, 2, 1)),
            ('unprojection_relu', nn.ReLU(inplace=True)),
        ]))

        # Output layer.
        self.prediction_dropout = nn.Dropout(prediction_dropout)
        self.prediction = \
            nn.Linear(bert_emb.embedding_dim, bert_emb.num_embeddings)
        self.prediction.weight = self.embedding.weight
        self.prediction.weight.requires_grad = False


class PervasiveEmbedding(Pervasive):
    """
    This class is a Pervasive neural network meant to be trained on embedding
    vectors using mean squared error (MSE). As such it lacks encoding/decoding
    layers, and its inputs and outputs have no interpretation as natural
    language tokens.
    """

    def __init__(self,
                 block_sizes,
                 bos,
                 Ts=51,
                 Tt=51,
                 initial_emb_size=768,
                 emb_size=128,
                 emb_dropout=0.2,
                 conv_dropout=0.2,
                 prediction_dropout=0.2,
                 division_factor=2,
                 growth_rate=32,
                 bias=False,
                 efficient=False):
        nn.Module.__init__(self)

        self.bos = bos
        self.Ts = Ts
        self.Tt = Tt

        # Embed and project to a lower-dimensional embedding.
        self.projection = nn.Sequential(OrderedDict([
            ('projection_dropout', nn.Dropout(emb_dropout)),
            ('projection', nn.Linear(initial_emb_size, emb_size)),
            ('projection_perm1', PermutationLayer(0, 2, 1)),
            ('projection_norm', nn.BatchNorm1d(emb_size)),
            ('projection_perm2', PermutationLayer(0, 2, 1)),
            ('projection_relu', nn.ReLU(inplace=True)),
        ]))

        # Source and target embeddings will be concatenated to form input.
        self.network = PervasiveNetwork(
            block_sizes, Ts, Tt, emb_size, conv_dropout,
            division_factor, growth_rate, bias, efficient)

        # Unprojection layer.
        self.unprojection = nn.Sequential(OrderedDict([
            ('unprojection_dropout', nn.Dropout(emb_dropout)),
            ('unprojection', nn.Linear(emb_size, initial_emb_size)),
            ('unprojection_perm1', PermutationLayer(0, 2, 1)),
            ('unprojection_norm', nn.BatchNorm1d(initial_emb_size)),
            ('unprojection_perm2', PermutationLayer(0, 2, 1)),
            ('unprojection_relu', nn.ReLU(inplace=True)),
        ]))

    def forward(self, data):
        """
        Forward pass of `data` through network in a training context.
        """
        Tt = data.shape[1] - self.Ts
        src_data, tgt_data = data.split([self.Ts, Tt], dim=1)
        X = self._forward(src_data, tgt_data)

        # No logits.

        # The target output does not have a BOS token, so it is one shorter
        # than the input. Drop the extraneous final token here.
        return X[:, :-1, :]

    def _forward(self, src_data, tgt_data):
        """
        Forward pass of `data` through the entire model.
        """
        # Apply projection and form 2D grid of pairs of word embeddings.
        src_emb = self.projection(src_data)
        tgt_emb = self.projection(tgt_data)

        # Run embeddings through the network.
        src_grid = src_emb.unsqueeze(1).repeat(1, tgt_emb.shape[1], 1, 1)
        tgt_grid = tgt_emb.unsqueeze(2).repeat(1, 1, src_emb.shape[1], 1)
        X = torch.cat((src_grid, tgt_grid), dim=3)
        X = X.permute(0, 3, 1, 2)

        # DenseNet.
        X = self.network(X)

        # Unprojection layer.
        X = self.unprojection(X)

        return X

    def predict(self, data):
        """
        Forward pass of `data` through the network for prediction. It
        should be of shape [batch_size, Ts + Tt]. The tgt_data it
        contains should have beginning of sequence (BOS) tokens.
        """
        Tt = data.shape[1] - self.Ts
        src_data, tgt_data = data.split([self.Ts, Tt], dim=1)
        # No logits.
        return self._forward(src_data, tgt_data)
