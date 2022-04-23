
import math
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data

class MyDataSet(Data.Dataset):
    def __init__(self, enc_inputs, dec_inputs, dec_outputs):
        super(MyDataSet, self).__init__()
        self.enc_inputs = enc_inputs
        self.dec_inputs = dec_inputs
        self.dec_outputs = dec_outputs

    def __len__(self):
        return self.enc_inputs.shape[0]

    def __getitem__(self, idx):
        return self.enc_inputs[idx], self.dec_inputs[idx], self.dec_outputs[idx]

# Transformer Model

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: [seq_len, batch_size, d_model]
        """
        x = x + self.pe[:x.size(0), :]   #pe:[max_len,1,d_model]
        return self.dropout(x)


def get_attn_pad_mask(seq_q, seq_k):

    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()

    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)
    return pad_attn_mask.expand(batch_size, len_q, len_k)


def get_attn_subsequence_mask(seq):
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequence_mask = np.triu(np.ones(attn_shape), k=1)
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()
    return subsequence_mask


# ==========================================================================================
class ScaledDotProductAttention(nn.Module):
    def __init__(self,d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k=d_k

    def forward(self, Q, K, V, attn_mask):
        """
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        """
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)  # scores : [batch_size, n_heads, len_q, len_k]
        scores.masked_fill_(attn_mask, -1e9)  # Fills elements of self tensor with value where mask is True.

        attn = nn.Softmax(dim=-1)(scores)
        # scores : [batch_size, n_heads, len_q, len_k] * V: [batch_size, n_heads, len_v(=len_k), d_v]
        context = torch.matmul(attn, V)  # context: [batch_size, n_heads, len_q, d_v]
        return context, attn


class MultiHeadAttention(nn.Module):

    def __init__(self,d_model, n_head,d_k, d_v, dropout, device):
        super(MultiHeadAttention, self).__init__()

        self.d_model = d_model
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.dropout = dropout
        self.device = device

        self.W_Q = nn.Linear(self.d_model, self.d_k * self.n_head, bias=False)
        self.W_K = nn.Linear(self.d_model, self.d_k * self.n_head, bias=False)
        self.W_V = nn.Linear(self.d_model, self.d_v * self.n_head, bias=False)
        self.attention = ScaledDotProductAttention(self.d_k)
        self.fc = nn.Linear(self.n_head * self.d_v, self.d_model, bias=False)

    def forward(self, input_Q, input_K, input_V, attn_mask):
        """
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        attn_mask: [batch_size, seq_len, seq_len]
        """
        residual, batch_size = input_Q, input_Q.size(0)
        # Q: [batch_size, n_heads, len_q, d_k]
        Q = self.W_Q(input_Q).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)
        # K: [batch_size, n_heads, len_k, d_k]
        K = self.W_K(input_K).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)
        # V: [batch_size, n_heads, len_v(=len_k), d_v]
        V = self.W_V(input_V).view(batch_size, -1, self.n_head, self.d_v).transpose(1, 2)

        # attn_mask: [batch_size, seq_len, seq_len] -> [batch_size, n_heads, seq_len, seq_len]
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_head, 1, 1)

        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        context, attn = self.attention(Q, K, V, attn_mask)
        # context: [batch_size, n_heads, len_q, d_v] -> [batch_size, len_q, n_heads * d_v]
        context = context.transpose(1, 2).reshape(batch_size, -1, self.n_head * self.d_v)
        output = self.fc(context)  # [batch_size, len_q, d_model]
        return nn.LayerNorm(self.d_model).to(self.device)(output + residual), attn


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self,d_model,d_inner,device):
        super(PoswiseFeedForwardNet, self).__init__()
        self.device = device
        self.d_model=d_model
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_inner, bias=False),
            nn.ReLU(),
            nn.Linear(d_inner, d_model, bias=False)
        )

    def forward(self, inputs):
        """
        inputs: [batch_size, seq_len, d_model]
        """
        residual = inputs
        output = self.fc(inputs)
        return nn.LayerNorm(self.d_model).to(self.device)(output + residual)  # [batch_size, seq_len, d_model]


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout, device):
        super(EncoderLayer, self).__init__()

        self.d_model = d_model
        self.d_inner = d_inner
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.dropout = dropout
        self.device = device
        self.enc_self_attn = MultiHeadAttention(d_model=self.d_model,n_head=self.n_head,
                                                d_k=self.d_k, d_v=self.d_v,
                                                dropout=self.dropout, device = self.device)
        self.pos_ffn = PoswiseFeedForwardNet(d_model=self.d_model,d_inner=self.d_inner,device=self.device)

    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs,
                                               enc_self_attn_mask)  # enc_inputs to same Q,K,V（未线性变换前）
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs, attn


class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout,device):
        super(DecoderLayer, self).__init__()

        self.d_model = d_model
        self.d_inner = d_inner
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.dropout = dropout
        self.device = device

        self.dec_self_attn = MultiHeadAttention(d_model=self.d_model,n_head=self.n_head,
                                                d_k=self.d_k, d_v=self.d_v,
                                                dropout=self.dropout,device=self.device)
        self.dec_enc_attn = MultiHeadAttention(d_model=self.d_model,n_head=self.n_head,
                                                d_k=self.d_k, d_v=self.d_v,
                                                dropout=self.dropout,device=self.device)
        self.pos_ffn = PoswiseFeedForwardNet(d_model=self.d_model,d_inner=self.d_inner,device=self.device)

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):

        # dec_outputs: [batch_size, tgt_len, d_model], dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len]
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs,
                                                        dec_self_attn_mask)
        # dec_outputs: [batch_size, tgt_len, d_model], dec_enc_attn: [batch_size, h_heads, tgt_len, src_len]
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs,
                                                      dec_enc_attn_mask)
        dec_outputs = self.pos_ffn(dec_outputs)  # [batch_size, tgt_len, d_model]
        return dec_outputs, dec_self_attn, dec_enc_attn


class Encoder(nn.Module):
    def __init__(self,d_model, d_inner, n_layers, n_head, d_k, d_v, dropout, n_position, device):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.d_inner = d_inner
        self.n_layers = n_layers
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.dropout = dropout
        self.n_position = n_position
        self.device = device
        self.pos_emb = PositionalEncoding(self.d_model, self.dropout, self.n_position)
        self.layers = nn.ModuleList([EncoderLayer(self.d_model, self.d_inner, self.n_head,
                                                  self.d_k, self.d_v, self.dropout, device=self.device) for _ in range(self.n_layers)])

    def forward(self, enc_inputs, enc_inputs_l):
        """
        enc_inputs: [batch_size, src_len]
        """
        # enc_outputs = self.src_emb(enc_inputs)  # [batch_size, src_len, d_model]
        enc_outputs = self.pos_emb(enc_inputs.transpose(0, 1)).transpose(0, 1)  # [batch_size, src_len, d_model]
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs_l, enc_inputs_l)  # [batch_size, src_len, src_len]
        enc_self_attns = []
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs,
                                               enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns


class Decoder(nn.Module):
    def __init__(self,d_model, d_inner, n_layers, n_head, d_k, d_v, dropout, n_position, device):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.d_inner = d_inner
        self.n_layers = n_layers
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.dropout = dropout
        self.n_position = n_position
        self.device = device

        self.pos_emb = PositionalEncoding(self.d_model, self.dropout, self.n_position)
        self.layers = nn.ModuleList([DecoderLayer(self.d_model, self.d_inner, self.n_head,
                                                  self.d_k, self.d_v, self.dropout,self.device) for _ in range(self.n_layers)])

    def forward(self, enc_outputs, dec_inputs, enc_inputs_l, dec_inputs_l):

        dec_outputs = self.pos_emb(dec_inputs.transpose(0, 1)).transpose(0, 1).to(
            self.device)
        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs_l, dec_inputs_l).to(self.device)
        dec_self_attn_subsequence_mask = get_attn_subsequence_mask(dec_inputs_l).to(
            self.device)

        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequence_mask),
                                      0).to(self.device)

        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs_l, enc_inputs_l)

        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:

            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs, dec_self_attn_mask,
                                                             dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        return dec_outputs, dec_self_attns, dec_enc_attns


class Transformer(nn.Module):
    def __init__(self, d_model=512, d_inner=2048,
            n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1, n_position=200, device='cpu'):
        super(Transformer, self).__init__()

        self.d_model = d_model
        self.d_inner = d_inner
        self.n_layers = n_layers
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.dropout = dropout
        self.n_position = n_position
        self.encoder = Encoder(d_model=d_model, d_inner=d_inner,n_layers=n_layers,
                               n_head=n_head, d_k=d_k, d_v=d_v, dropout=dropout,
                               n_position=n_position,device=device).to(device)
        self.decoder = Decoder(d_model=d_model, d_inner=d_inner,n_layers=n_layers,
                               n_head=n_head, d_k=d_k, d_v=d_v, dropout=dropout,
                               n_position=n_position,device=device).to(device)
        self.projection = nn.Linear(d_model, d_model, bias=False).to(device)

    def forward(self, enc_inputs, dec_inputs, enc_inputs_l, dec_inputs_l):
        enc_outputs, enc_self_attns = self.encoder(enc_inputs, enc_inputs_l)
        dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(enc_outputs, dec_inputs, enc_inputs_l, dec_inputs_l)
        dec_logits = self.projection(dec_outputs)
        return dec_logits.view(-1, dec_logits.size(-1)), enc_self_attns, dec_self_attns, dec_enc_attns

