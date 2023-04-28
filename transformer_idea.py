import torch 
import torch.nn as nn 
import torch.nn.functional as F
from typing import Optional, Any
from torch import Tensor
import pdb 
import copy 

# class TransformerDecoderLayer(nn.Module):
#     r"""TransformerDecoderLayer is made up of self-attn, multi-head-attn and feedforward network.
#     This standard decoder layer is based on the paper "Attention Is All You Need".
#     Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
#     Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
#     Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
#     in a different way during application.

#     Args:
#         d_model: the number of expected features in the input (required).
#         nhead: the number of heads in the multiheadattention models (required).
#         dim_feedforward: the dimension of the feedforward network model (default=2048).
#         dropout: the dropout value (default=0.1).
#         activation: the activation function of intermediate layer, relu or gelu (default=relu).

#     Examples::
#         >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
#         >>> memory = torch.rand(10, 32, 512)
#         >>> tgt = torch.rand(20, 32, 512)
#         >>> out = decoder_layer(tgt, memory)
#     """

#     def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
#         super(TransformerDecoderLayer, self).__init__()
#         self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
#         self.multihead_attn_audio = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
#         self.multihead_attn_tag = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
#         self.multihead_attn_t_to_a = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

#         # Implementation of Feedforward model
#         self.linear1 = nn.Linear(d_model, dim_feedforward)
#         self.dropout = nn.Dropout(dropout)
#         self.linear2 = nn.Linear(dim_feedforward, d_model)

#         self.norm1 = nn.LayerNorm(d_model)
#         self.norm2 = nn.LayerNorm(d_model)
#         self.norm3 = nn.LayerNorm(d_model)
#         self.dropout1 = nn.Dropout(dropout)
#         self.dropout2 = nn.Dropout(dropout)
#         self.dropout3 = nn.Dropout(dropout)
#         self.dropout4 = nn.Dropout(dropout)
        
#         self.activation = _get_activation_fn(activation)

#         self.linear_at1 = nn.Linear(d_model*2, d_model*2*4)
#         self.dropout_at = nn.Dropout(dropout)
#         self.linear_at2 = nn.Linear(d_model*2*4, d_model)
#     def __setstate__(self, state):
#         if 'activation' not in state:
#             state['activation'] = F.relu
#         super(TransformerDecoderLayer, self).__setstate__(state)

#     def forward(self, tgt: Tensor, memory_a: Tensor, memory_t: Tensor, tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None,
#                 tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
#         r"""Pass the inputs (and mask) through the decoder layer.

#         Args:
#             tgt: the sequence to the decoder layer (required).
#             memory: the sequence from the last layer of the encoder (required).
#             tgt_mask: the mask for the tgt sequence (optional).
#             memory_mask: the mask for the memory sequence (optional).
#             tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
#             memory_key_padding_mask: the mask for the memory keys per batch (optional).

#         Shape:
#             see the docs in Transformer class.
#         """
#         tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
#                               key_padding_mask=tgt_key_padding_mask)[0]
#         tgt = tgt + self.dropout1(tgt2)
#         tgt = self.norm1(tgt)
#         # pdb.set_trace()
#         tgt2 = self.multihead_attn_audio(tgt, memory_a, memory_a, attn_mask=memory_mask,
#                                    key_padding_mask=memory_key_padding_mask)[0]
#         tgt2_t = self.multihead_attn_tag(tgt, memory_t, memory_t, attn_mask=memory_mask,
#                                    key_padding_mask=memory_key_padding_mask)[0]
        
#         tgt2_t2a = self.multihead_attn_t_to_a(tgt2_t, memory_a, memory_a, attn_mask=memory_mask,
#                                    key_padding_mask=memory_key_padding_mask)[0]
#         # pdb.set_trace()

#         # tgt = tgt + self.dropout2(tgt2) + self.dropout4(tgt2_t2a)
#         tgt = tgt + self.dropout4(self.linear_at2(self.dropout_at(self.activation(self.linear_at1(torch.cat((tgt2,tgt2_t2a),dim=-1))))))
#         tgt = self.norm2(tgt)
#         tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
#         tgt = tgt + self.dropout3(tgt2)
#         tgt = self.norm3(tgt)
#         return tgt
    

class TransformerDecoder(nn.Module):
    r"""TransformerDecoder is a stack of N decoder layers

    Args:
        decoder_layer: an instance of the TransformerDecoderLayer() class (required).
        num_layers: the number of sub-decoder-layers in the decoder (required).
        norm: the layer normalization component (optional).

    Examples::
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        >>> transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        >>> memory = torch.rand(10, 32, 512)
        >>> tgt = torch.rand(20, 32, 512)
        >>> out = transformer_decoder(tgt, memory)
    """
    __constants__ = ['norm']

    def __init__(self, decoder_layer, num_layers, norm=None):
        super(TransformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt: Tensor, memory_a: Tensor, memory_t: Tensor, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                text_key_padding_mask: Optional[Tensor] = None, return_weights=False) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer in turn.

        Args:
            tgt: the sequence to the decoder (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = tgt
        all_attn_weights = []
        all_attn_weights_words = []
        for mod in self.layers:
            output, attn_weights, attn_weights_words = mod(output, memory_a, memory_t, tgt_mask=tgt_mask,
                         memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask,
                         text_key_padding_mask=text_key_padding_mask)
            all_attn_weights.append(attn_weights.detach().cpu())
            all_attn_weights_words.append(attn_weights_words.detach().cpu())
        if self.norm is not None:
            output = self.norm(output)

        # return output
        if return_weights:
            # return output,all_attn_weights[-1]
            # pdb.set_trace()
            # all_attn_weights = torch.stack(all_attn_weights).mean(dim=0)
            # return output, all_attn_weights
            return output, all_attn_weights, all_attn_weights_words
        else:
            return output

## 0.27 sample
# class TransformerDecoderLayer(nn.Module):
#     r"""TransformerDecoderLayer is made up of self-attn, multi-head-attn and feedforward network.
#     This standard decoder layer is based on the paper "Attention Is All You Need".
#     Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
#     Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
#     Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
#     in a different way during application.

#     Args:
#         d_model: the number of expected features in the input (required).
#         nhead: the number of heads in the multiheadattention models (required).
#         dim_feedforward: the dimension of the feedforward network model (default=2048).
#         dropout: the dropout value (default=0.1).
#         activation: the activation function of intermediate layer, relu or gelu (default=relu).

#     Examples::
#         >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
#         >>> memory = torch.rand(10, 32, 512)
#         >>> tgt = torch.rand(20, 32, 512)
#         >>> out = decoder_layer(tgt, memory)
#     """

#     def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
#         super(TransformerDecoderLayer, self).__init__()
#         self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
#         self.multihead_attn_audio = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
#         self.multihead_attn_tag = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        
#         self.res_layer_cap = ResidualConnection(d_model, dropout)
#         self.res_layer_audio = ResidualConnection(d_model, dropout)
#         self.res_layer_tag = ResidualConnection(d_model, dropout)
#         # Implementation of Feedforward model
#         self.linear1 = nn.Linear(d_model, dim_feedforward)
#         self.dropout = nn.Dropout(dropout)
#         self.linear2 = nn.Linear(dim_feedforward, d_model)
#         self.norm3 = nn.LayerNorm(d_model)
#         self.dropout3 = nn.Dropout(dropout)
        
#         #####
#         self.activation = _get_activation_fn(activation)

#         self.norm_ca = nn.LayerNorm(d_model)
#         self.norm_ct = nn.LayerNorm(d_model)
#         self.a_t_constant = nn.Parameter(torch.tensor([0.0]))

#         self.norm_ = nn.LayerNorm(d_model)

#         # gated 
#         self.linear_a = nn.Linear(d_model, d_model)

#         self.linear_t = nn.Linear(d_model, d_model)
#     def __setstate__(self, state):
#         if 'activation' not in state:
#             state['activation'] = F.relu
#         super(TransformerDecoderLayer, self).__setstate__(state)

#     def forward(self, tgt: Tensor, memory_a: Tensor, memory_t: Tensor, tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None,
#                 tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None,
#                 text_key_padding_mask: Optional[Tensor] = None) -> Tensor:
#         r"""Pass the inputs (and mask) through the decoder layer.

#         Args:
#             tgt: the sequence to the decoder layer (required).
#             memory: the sequence from the last layer of the encoder (required).
#             tgt_mask: the mask for the tgt sequence (optional).
#             memory_mask: the mask for the memory sequence (optional).
#             tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
#             memory_key_padding_mask: the mask for the memory keys per batch (optional).

#         Shape:
#             see the docs in Transformer class.
#         """

#         def sublayer_self_att(C): return self.self_attn(C, C, C, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
#         def sublayer_enc_att_A(C): return self.multihead_attn_audio(C, memory_a, memory_a, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)[0]
#         def sublayer_enc_att_T(C): return self.multihead_attn_tag(C, memory_t, memory_t, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)[0]

#         # pdb.set_trace()
#         tgt = self.res_layer_cap(tgt, sublayer_self_att)
#         Ca = self.res_layer_audio(tgt, sublayer_enc_att_A)
#         Ct = self.res_layer_audio(tgt, sublayer_enc_att_T)

#         # Ca = self.norm_ca(Ca)
#         # Ct = self.norm_ct(Ct)

#         av_factor = torch.sigmoid(torch.clamp(self.a_t_constant, min=-3, max=3))
#         fused_features = av_factor * Ct + (1 - av_factor) * Ca
#         # add gated 
#         # gate_a = torch.sigmoid(self.linear_a(tgt))
#         # gate_t = torch.sigmoid(self.linear_t(tgt))
#         # fused_features = gate_a * Ca +  gate_t * Ca
#         ###
#         # fused_features = Ct +  Ca

#         fused_features = self.norm_(fused_features)
#         # pdb.set_trace()
#         ## new  
#         tgt2 = self.linear2(self.dropout(self.activation(self.linear1(fused_features))))
#         tgt = tgt + self.dropout3(tgt2)
#         tgt = self.norm3(tgt)

#         return fused_features
    

class TransformerDecoderLayer(nn.Module):
    r"""TransformerDecoderLayer is made up of self-attn, multi-head-attn and feedforward network.
    This standard decoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).

    Examples::
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        >>> memory = torch.rand(10, 32, 512)
        >>> tgt = torch.rand(20, 32, 512)
        >>> out = decoder_layer(tgt, memory)
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn_audio = nn.MultiheadAttention(d_model, nhead, dropout=0)
        self.multihead_attn_tag = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        
        self.res_layer_cap = ResidualConnection(d_model, dropout,False)
        self.res_layer_audio = ResidualConnection(d_model, dropout,False)
        self.res_layer_tag = ResidualConnection(d_model, dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(dropout)
        
        #####
        self.activation = _get_activation_fn(activation)

        self.norm_ca = nn.LayerNorm(d_model)
        self.norm_ct = nn.LayerNorm(d_model)
        self.a_t_constant = nn.Parameter(torch.tensor([0.0]))

        self.norm_ = nn.LayerNorm(d_model)

        # gated 
        # self.linear_a = nn.Linear(d_model, d_model)

        # self.linear_t = nn.Linear(d_model, d_model)
    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerDecoderLayer, self).__setstate__(state)

    def forward(self, tgt: Tensor, memory_a: Tensor, memory_t: Tensor, tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None,
                text_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """

        def sublayer_self_att(C): return self.self_attn(C, C, C, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)
        def sublayer_enc_att_A(C): return self.multihead_attn_audio(C, memory_a, memory_a, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask) # [0]
        def sublayer_enc_att_T(C): return self.multihead_attn_tag(C, memory_t, memory_t, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)

        # pdb.set_trace()
        tgt = self.res_layer_cap(tgt, sublayer_self_att)
        Ca, attn_weights = self.res_layer_audio(tgt, sublayer_enc_att_A, True)
        Ct, attn_weights_words = self.res_layer_audio(tgt, sublayer_enc_att_T, True)
        # Ct = self.res_layer_audio(tgt, sublayer_enc_att_T)


        # Ca = self.norm_ca(Ca)
        # Ct = self.norm_ct(Ct)

        av_factor = torch.sigmoid(torch.clamp(self.a_t_constant, min=-3, max=3))
        fused_features = av_factor * Ct + (1 - av_factor) * Ca
        # add gated 
        # gate_a = torch.sigmoid(self.linear_a(tgt))
        # gate_t = torch.sigmoid(self.linear_t(tgt))
        # fused_features = (1 - av_factor)  * gate_a * Ca +  av_factor * gate_t * Ct
        ###
        # fused_features = Ct +  Ca
        ####### 27.4
        # fused_features = self.norm_(fused_features)
        # ## new  
        # tgt2 = self.linear2(self.dropout(self.activation(self.linear1(fused_features))))
        # tgt = tgt + self.dropout3(tgt2)
        # tgt = self.norm3(tgt)

        # return fused_features


        # cuda_0
        fused_features = self.norm_(fused_features)
        ## new  
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(fused_features))))
        fused_feature_t = fused_features + self.dropout3(tgt2)
        fused_feature_t = self.norm3(fused_feature_t)

        return fused_feature_t, attn_weights, attn_weights_words


# class ResidualConnection(nn.Module):

#     def __init__(self, size, dout_p):
#         super(ResidualConnection, self).__init__()
#         self.norm = nn.LayerNorm(size)
#         self.dropout = nn.Dropout(dout_p)

#     def forward(self, x, sublayer):  
#         # x (B, S, D)
#         res = self.norm(x)
#         res = sublayer(res)
#         res = self.dropout(res)

#         return x + res
    
class ResidualConnection(nn.Module):

    def __init__(self, size, dout_p, use_norm=True):
        super(ResidualConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dout_p)
        self.use_norm = use_norm
    def forward(self, x, sublayer, return_attn=False):  
        # x (B, S, D)
        x_, att = sublayer(x)
        res = self.dropout(x_)
        x = x + res 
        if self.use_norm:
            x = self.norm(x)
        else:
            pass
        
        if return_attn:
            return x, att
        else:
            return x
    

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])



def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))
