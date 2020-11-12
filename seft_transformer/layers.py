import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from einops import rearrange, reduce
import numpy as np
import sys
import pdb

from .blocks import *


class inpLayer(layers.Layer):
  def __init__(self, enc_dim=128):
    super(inpLayer, self).__init__()
    self.posEncoding = posEncodingBlock(enc_dim=enc_dim)
    self.inpEncoding = inpEncodingBlock(enc_dim=enc_dim)

  def call(self, inp, time, mask):
    """
    Input shapes:
      inp:  (b, t, m, i)
      time: (b, t)
      mask: (b, t, m)
    Output shapes:
      return: (b, t, m, d)
    """
    pos_enc = self.posEncoding(time, mask) # (b, t, m, d)
    inp_enc = self.inpEncoding(inp) # (b, t, m, d)
    tot_enc = inp_enc + pos_enc # (b, t, m, d)
    return tot_enc

class encLayer(layers.Layer):
  def __init__(self, proj_dim=128, enc_dim=128, num_head=4, ff_dim=128):
    super(encLayer, self).__init__()
    self.axAttention = axialMultiHeadAttentionBlock(
        proj_dim=proj_dim, 
        enc_dim=enc_dim,
        num_head=num_head
    )
    self.posFeedforward = posFeedforwardBlock(
        enc_dim=enc_dim, 
        ff_dim=ff_dim
    )
    self.layerNorm1 = layers.LayerNormalization()
    self.layerNorm2 = layers.LayerNormalization()

  def call(self, inp, mask):
    """
    Input shapes:
      inp:  (b, t, m, d)
      mask: (b, t, m)
    Output shapes:
      return: (b, t, m, d)
    """
    # Project query, key and value
    attn = self.axAttention(inp, mask) # (b, t, m, d)
    attn_out = self.layerNorm1(inp + attn) # (b, t, m, d)
    ffn = self.posFeedforward(attn_out) # (b, t, m, d)
    ffn_out = self.layerNorm2(attn_out + ffn) # (b, t, m, d)
    return ffn_out

class predLayer(layers.Layer):
  def __init__(self, ff_dim=32):
    super(predLayer, self).__init__()
    self.ff_dim = ff_dim
  
  def build(self, input_shape):
    # Dense layer to aggregate different modalities
    self.denseMod = layers.Dense(1, activation='relu')
    # Dense layers to predict classes
    self.densePred1 = layers.Dense(self.ff_dim, activation='relu')
    self.densePred2 = layers.Dense(1, activation='sigmoid')

  def call(self, inp):
    """
    Input shapes:
      inp:  (b, t, m, d)
    Output shapes:
      return: (b, 1)
    """
    out = reduce(inp, 'b t m d -> b d m', 'mean')
    out = self.denseMod(out) # (b, d, 1)
    out = rearrange(out, 'b d 1 -> b d') # (b, d)
    pred = self.densePred2(self.densePred1(out)) # (b, 1)
    return pred

