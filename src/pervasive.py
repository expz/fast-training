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
from pytorch_pretrained_bert import BertTokenizer, BertModel
import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp

from dataloader import VocabData


# Cache most recent function result using @cache decorator.
cache = functools.lru_cache(1)


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
        return super(MaskedConv2d, self).forward(x)


class DenseLayer(nn.Module):
    """
    Layer of a DenseNet.
    
    Adapted from https://github.com/elbayadm/attn2d/blob/
                         master/nmt/models/efficient_densenet.py
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

    def __init__(self,
                 num_layers,
                 input_size,
                 bn_size,
                 growth_rate,
                 dropout,
                 bias,
                 efficient=False):
        super(DenseBlock, self).__init__()
        kernel_size = 3
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
                 efficient=False):
        self.Ts = Ts
        self.Tt = Tt
        self.input_channels = 2 * emb_size

        densenet = DenseNet(self.input_channels,
                            block_sizes,
                            dropout=conv_dropout,
                            growth_rate=growth_rate,
                            division_factor=division_factor,
                            bias=bias,
                            efficient=efficient)

        aggregator = Aggregator()

        linear = nn.Linear(densenet.output_channels, emb_size)
        relu = nn.ReLU(inplace=True)

        super().__init__(OrderedDict([
            ('densenet', densenet),
            ('aggregator', aggregator),
            ('linear', linear),
            ('relu', relu)
        ]))


class PermutationLayer(nn.Module):
    """
    This class allows performing permutations in modules defined using the
    nn.Sequential constructor.
    """

    def __init__(self, *dims):
        nn.Module.__init__(self)
        self.dims = dims

    def forward(self, X):
        return X.permute(*dims)


class Pervasive(nn.Module):
    """
    Pervasive Attention Network.
    
    This is a reimplementation of the model in
    https://github.com/elbayadm/attn2d/
            blob/master/nmt/models/pervasive.py
    """

    def __init__(self,
                 block_sizes,
                 src_vocab_sz,
                 tgt_vocab_sz,
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
                 efficient=False):
        nn.Module.__init__(self)

        self.bos = bos
        self.Ts = Ts
        self.Tt = Tt

        self.emb_dropout = nn.Dropout(emb_dropout)
        self.src_embedding = nn.Embedding(src_vocab_sz, emb_size)
        self.tgt_embedding = nn.Embedding(tgt_vocab_sz, emb_size)

        self.network = PervasiveNetwork(
            block_sizes, Ts, Tt, emb_size, conv_dropout, divison_factor,
            growth_rate, bias, efficient)

        self.prediction_dropout = nn.Dropout(prediction_dropout)
        self.prediction = nn.Linear(emb_size, tgt_vocab_sz)
        self.prediction.weight = self.tgt_embedding.weight

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

        # The target output does not have a BOS token, so it is one shorter
        # than the input. Drop the extraneous final token here.
        logits = F.log_softmax(X, dim=2)[:, :-1]
        return logits.permute(0, 2, 1)

    def _forward(self, src_data, tgt_data):
        """
        Forward pass of `data` through the entire model.
        """
        # Embedding.
        src_emb = self.emb_dropout(self.src_embedding(src_data))
        tgt_emb = self.emb_dropout(self.tgt_embedding(tgt_data))
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


class PervasiveBert(Pervasive):
    """
    This class is a Pervasive neural network which uses BERT pre-trained
    multilingual, cased embeddings. The embeddings are 
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
                 efficient=False):
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
            ('projection', nn.Linear(self.embedding.embedding_dim, emb_size)),
            #('projection_perm1', PermutationLayer(0, 2, 1)),
            #('projection_norm', nn.BatchNorm1d(emb_size)),
            #('projection_perm2', PermutationLayer(0, 2, 1)),
            ('projection_relu', nn.ReLU(inplace=True)),
        ]))

        # Source and target embeddings will be concatenated to form input.
        self.network = PervasiveNetwork(
            block_sizes, Ts, Tt, emb_size, conv_dropout, division_factor,
            growth_rate, bias, efficient)

        # Output layer.
        self.unprojection_dropout = nn.Dropout(emb_dropout)
        self.unprojection = nn.Linear(emb_size, bert_emb.embedding_dim)
        self.unprojection_norm = nn.BatchNorm1d(bert_emb.embedding_dim)
        self.unprojection_relu = nn.ReLU(inplace=True)
        self.prediction_dropout = nn.Dropout(prediction_dropout)
        self.prediction = nn.Linear(bert_emb.embedding_dim, bert_emb.num_embeddings)
        self.prediction.weight = self.embedding.weight
        self.prediction.weight.requires_grad = False

    def _forward(self, src_data, tgt_data):
        """
        Forward pass of `data` through the network.
        """
        # Embedding with projection.
        src_emb = self.projection(self.embedding(src_data))
        tgt_emb = self.projection(self.embedding(tgt_data))
        src_grid = src_emb.unsqueeze(1).repeat(1, tgt_emb.shape[1], 1, 1)
        tgt_grid = tgt_emb.unsqueeze(2).repeat(1, 1, src_emb.shape[1], 1)
        X = torch.cat((src_grid, tgt_grid), dim=3)

        # Embedding dim becomes channels.
        X = X.permute(0, 3, 1, 2)

        # Pass data through DenseNet.
        X = self.network(X)

        # Unprojection layer.
        X = self.unprojection(self.unprojection_dropout(X))
        X = X.permute(0, 2, 1)
        X = self.unprojection_norm(X)
        X = X.permute(0, 2, 1)
        X = self.unprojection_relu(X)

        X = self.prediction(self.prediction_dropout(X))
        return X

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
                 input_size=768,
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
            ('projection', nn.Linear(input_size, emb_size)),
            ('projection_relu', nn.ReLU(inplace=True)),
        ]))

        # Source and target embeddings will be concatenated to form input.
        self.network = PervasiveNetwork(
            block_sizes, Ts, Tt, emb_size, conv_dropout,
            division_factor, growth_rate, bias, efficient)

        # Output layer.
        self.unprojection_dropout = nn.Dropout(emb_dropout)
        self.unprojection = nn.Linear(emb_size, input_size)
        self.unprojection_norm = nn.BatchNorm1d(input_size)
        self.unprojection_relu = nn.ReLU(inplace=True)

    def forward(self, data):
        """
        Forward pass of `data` through network in a training context.
        """
        Tt = data.shape[1] - self.Ts
        src_data, tgt_data = data.split([self.Ts, Tt], dim=1)
        X = self._forward(src_data, tgt_data)
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
        src_grid = src_emb.unsqueeze(1).repeat(1, tgt_emb.shape[1], 1, 1)
        tgt_grid = tgt_emb.unsqueeze(2).repeat(1, 1, src_emb.shape[1], 1)
        X = torch.cat((src_grid, tgt_grid), dim=3)

        # Run embeddings through the network.
        X = X.permute(0, 3, 1, 2)
        X = self.network(X)

        # Unprojection layer.
        X = self.unprojection(self.unprojection_dropout(X))
        X = X.permute(0, 2, 1)
        X = self.unprojection_norm(X)
        X = X.permute(0, 2, 1)
        X = self.unprojection_relu(X)

        return X

    def predict(self, data):
        """
        Forward pass of `data` through the network for prediction. It
        should be of shape [batch_size, Ts + Tt]. The tgt_data it
        contains should have beginning of sequence (BOS) tokens.
        """
        Tt = data.shape[1] - self.Ts
        src_data, tgt_data = data.split([self.Ts, Tt], dim=1)
        return self._forward(src_data, tgt_data)
