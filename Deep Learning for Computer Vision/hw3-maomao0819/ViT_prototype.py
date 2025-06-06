import torch
import torch.nn as nn
from torch.nn.functional import log_softmax, pad
import math
import copy


import warnings
from torch.nn.parallel import DistributedDataParallel as DDP

from utils import layer_debug_log

# Set to False to skip notebook execution (e.g. for debugging)
warnings.filterwarnings("ignore")
RUN_EXAMPLES = True


def show_example(fn, args=[]):
    if __name__ == "__main__" and RUN_EXAMPLES:
        return fn(*args)


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        # src: [batch_size, seqlen]
        # src_mask: [batch_size, 1, seqlen]
        # src_embed: [batch_size, seqlen, d_model]

        # feature: [batch_size, seqlen, d_model]

        src_embed = self.src_embed(src)
        feature = self.encoder(src_embed, src_mask)

        return feature

    def decode(self, memory, src_mask, tgt, tgt_mask):
        # memory: [batch_size, seqlen, d_model]
        # src_mask: [batch_size, 1, seqlen]
        # tgt: [batch_size, n_out_token]
        # tgt_mask: [batch_size, n_out_token, n_out_token]
        # tgt_embed: [batch_size, n_out_token, d_model]
        # out_token: [batch_size, n_out_token, d_model]

        tgt_embed = self.tgt_embed(tgt)
        out_token = self.decoder(tgt_embed, memory, src_mask, tgt_mask)
        return out_token


class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        # vocab: tgt_vocab
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        # x: [batch_size, d_model]

        # proj: [batch_size, tgt_vocab]
        # generate: [batch_size, tgt_vocab]

        proj = self.proj(x)
        generate = log_softmax(proj, dim=-1)
        return generate


def clones(module, n_layers):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n_layers)])


class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, n_layers):
        super(Encoder, self).__init__()
        self.layers = clones(layer, n_layers)
        self.norm = LayerNorm(layer.size)
        # layer.size: d_model

    def forward(self, x, mask):
        # x: [batch_size, seqlen, d_model]
        # mask: [batch_size, 1, seqlen]

        # hidden: [batch_size, seqlen, d_model]
        # feature: [batch_size, seqlen, d_model]

        "Pass the input (and mask) through each layer in turn."
        hidden = x
        for layer in self.layers:
            hidden = layer(hidden, mask)
        feature = self.norm(hidden)
        return feature


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, n_features, eps=1e-6):
        super(LayerNorm, self).__init__()
        # n_features: d_model
        self.a_2 = nn.Parameter(torch.ones(n_features))
        self.b_2 = nn.Parameter(torch.zeros(n_features))
        self.eps = eps

    def forward(self, x):
        # x:
        #     encode: [batch_size, seqlen, d_model]
        #     decode: [batch_size, n_out_token, d_model]

        # mean:
        #     encode: [batch_size, seqlen, 1]
        #     decode: [batch_size, n_out_token, 1]
        # std:
        #     encode: [batch_size, seqlen, 1]
        #     decode: [batch_size, n_out_token, 1]
        # norm:
        #     encode: [batch_size, seqlen, d_model]
        #     decode: [batch_size, n_out_token, d_model]

        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        norm = self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        return norm


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        # encode: [batch_size, seqlen, d_model]
        # decode: [batch_size, n_out_token, d_model]
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size
        # size: d_model

    def forward(self, x, mask):
        # x: [batch_size, seqlen, d_model]
        # mask: [batch_size, 1, seqlen]

        # attention_feature: [batch_size, seqlen, d_model]
        # feature: [batch_size, seqlen, d_model]

        attention_feature = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        feature = self.sublayer[1](attention_feature, self.feed_forward)
        return feature


class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, n_layers):
        super(Decoder, self).__init__()
        self.layers = clones(layer, n_layers)
        self.norm = LayerNorm(layer.size)
        # layer.size: d_model

    def forward(self, x, memory, src_mask, tgt_mask):
        # x: [batch_size, n_out_token, d_model]
        # memory: [batch_size, seqlen, d_model]
        # src_mask: [batch_size, 1, seqlen]
        # tgt_mask: [batch_size, n_out_token, n_out_token]

        # hidden: [batch_size, n_out_token, d_model]
        # out_token: [batch_size, n_out_token, d_model]

        hidden = x
        for layer in self.layers:
            hidden = layer(hidden, memory, src_mask, tgt_mask)
        out_token = self.norm(hidden)
        return out_token


class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, cross_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.cross_attn = cross_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        # x: [batch_size, n_out_token, d_model]
        # memory: [batch_size, seqlen, d_model]
        # src_mask: [batch_size, 1, seqlen]
        # tgt_mask: [1, n_out_token, n_out_token]

        # self_attn_feature: [batch_size, n_out_token, d_model]
        # cross_attn_feature: [batch_size, n_out_token, d_model]
        # feature: [batch_size, n_out_token, d_model]

        m = memory
        self_attn_feature = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        cross_attn_feature = self.sublayer[1](self_attn_feature, lambda x: self.cross_attn(x, m, m, src_mask))
        feature = self.sublayer[2](cross_attn_feature, self.feed_forward)
        return feature


def subsequent_mask(size):
    # size: n_out_token
    # subsequent_mask: [batch_size, n_out_token, n_out_token]
    # preceding_mask : [batch_size, n_out_token, n_out_token]
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8)
    preceding_mask = subsequent_mask == 0
    return preceding_mask


def attention(query, key, value, mask=None, dropout=None):
    
    # query:
    #     encode: [batch_size, n_heads, seqlen, d_k]
    #     decode: [batch_size, n_heads, n_out_token, d_k]
    # key, value:
    #     encode: [batch_size, n_heads, seqlen, d_k]
    #     decode-self_attn: [batch_size, n_heads, n_out_token, d_k]
    #     decode-cross_attn: [batch_size, n_heads, seqlen, d_k]
    # mask:
    #     encode: [batch_size, 1, 1, seqlen]
    #     decode-self_attn: [1, 1, n_out_token, n_out_token]
    #     decode-cross_attn: [batch_size, 1, 1, seqlen]

    # scores, p_attn:
    #     encode: [batch_size, n_heads, seqlen, seqlen]
    #     decode-self_attn: [batch_size, n_heads, n_out_token, n_out_token]
    #     decode-cross_attn: [batch_size, n_heads, n_out_token, seqlen]
    # attn:
    #     encode: [batch_size, n_heads, seqlen, d_k]
    #     decode: [batch_size, n_heads, n_out_token, d_k]

    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = scores.softmax(dim=-1)

    if dropout is not None:
        p_attn = dropout(p_attn)

    attn = torch.matmul(p_attn, value)

    return attn, p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, n_heads, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % n_heads == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        # query:
        #     encode: [batch_size, seqlen, d_model]
        #     decode: [batch_size, n_out_token, d_model]
        # key, value:
        #     encode: [batch_size, seqlen, d_model]
        #     decode-self_attn: [batch_size, n_out_token, d_model]
        #     decode-cross_attn: [batch_size, seqlen, d_model]
        # mask:
        #     encode: [batch_size, 1, seqlen]
        #     decode-self_attn: [1, n_out_token, n_out_token]
        #     decode-cross_attn: [batch_size, 1, seqlen]
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        n_batches = query.size(0)
        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [
            lin(x).view(n_batches, -1, self.n_heads, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]

        # query:
        #     encode: [batch_size, n_heads, seqlen, d_k]
        #     decode: [batch_size, n_heads, n_out_token, d_k]
        # key, value:
        #     encode: [batch_size, n_heads, seqlen, d_k]
        #     decode-self_attn: [batch_size, n_heads, n_out_token, d_k]
        #     decode-cross_attn: [batch_size, n_heads, seqlen, d_k]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # x:
        #     encode: [batch_size, n_heads, seqlen, d_k]
        #     decode: [batch_size, n_heads, n_out_token, d_k]
        # attn:
        #     encode: [batch_size, n_heads, seqlen, seqlen]
        #     decode-self_attn: [batch_size, n_heads, n_out_token, n_out_token]
        #     decode-cross_attn: [batch_size, n_heads, n_out_token, seqlen]

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(n_batches, -1, self.n_heads * self.d_k)

        # x:
        #     encode: [batch_size, seqlen, d_model]
        #     decode: [batch_size, n_out_token, d_model]

        del query
        del key
        del value
        x = self.linears[-1](x)
        return x


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # [batch_size, n_out_token, d_model]
        return self.w_2(self.dropout(self.w_1(x).relu()))


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        # x : [batch_size, n_out_token]

        # lut : [batch_size, n_out_token, d_model]

        lut = self.lut(x)
        return lut * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x:
        #     encode: [batch_size, seqlen, d_model]
        #     decode: [batch_size, n_out_token, d_model]

        # pe:
        #     encode: [batch_size, seqlen, d_model]
        #     decode: [batch_size, n_out_token, d_model]

        pe = self.pe[:, : x.size(1)].requires_grad_(False)
        pe = x + pe
        pe = self.dropout(pe)
        return pe


def make_model(src_vocab, tgt_vocab, n_layers=6, d_model=512, d_ff=2048, n_heads=8, dropout=0.1):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(n_heads, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), n_layers),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), n_layers),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab),
    )

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


def inference_test():
    src_vocab = 15
    tgt_vocab = 20
    N = 2
    # src: [batch_size, seqlen]
    # src_mask: [batch_size, 1, seqlen]
    test_model = make_model(src_vocab, tgt_vocab, N)
    test_model.eval()

    src = torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 10, 2]])
    src = torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 10, 2], [1, 2, 3, 4, 7, 5, 7, 8, 9, 10, 11, 10, 2]])
    batch_size = src.size(0)
    seqlen = src.size(-1)

    src_mask = torch.ones(batch_size, 1, seqlen)
    src_mask = (src != 0).unsqueeze(-2)

    memory = test_model.encode(src, src_mask)
    # memory: [batch_size, seqlen, d_model]

    ys = torch.zeros(batch_size, 1).type_as(src)
    for _ in range(9):
        # out: [batch_size, n_out_token, d_model]
        # prob: [batch_size, tgt_vocab]
        # next_word [1]
        # ys: [batch_size, n_out_token]
        out = test_model.decode(memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data))
        prob = test_model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat([ys, torch.empty(batch_size, 1).type_as(src.data).fill_(next_word)], dim=1)

    print("Example Untrained Model Prediction:", ys)


def run_tests():
    for _ in range(5):
        inference_test()


show_example(run_tests)
