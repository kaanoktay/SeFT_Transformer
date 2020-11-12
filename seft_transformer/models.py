import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from einops import rearrange
import numpy as np
import sys
import pdb

from .layers import *

class timeSeriesTransformer(keras.Model):
  def __init__(self, proj_dim=128, num_head=4, enc_dim=128, pos_ff_dim=128, pred_ff_dim=32):
    super(timeSeriesTransformer, self).__init__()
    self.inpEncoding = inpLayer(
        enc_dim=enc_dim
    )
    self.transformerEncoder = encLayer(proj_dim=proj_dim, 
        enc_dim=enc_dim, 
        num_head=num_head, 
        ff_dim=pos_ff_dim
    )
    self.predNet = predLayer(ff_dim=pred_ff_dim)

  def call(self, inputs):
    """
    Input shapes:
      data: a tuple of measurements, timestamps, masks etc.
    Output shapes:
      return: prediction
    """
    # Get inputs
    time = inputs[1] # (b, t)
    inp  = inputs[2] # (b, t, m)
    mask = inputs[3] # (b, t, m)
    # Expand input dimensions if necessary
    if len(inp.shape)==3:
      inp = rearrange(inp, 'b t m -> b t m 1')
    # Encode inputs
    enc_inp = self.inpEncoding(inp, time, mask) # (b, t, m, d)
    # Calculate attention
    attn = self.transformerEncoder(enc_inp, mask) # (b, t, m, d)
    # Make prediction
    pred = self.predNet(attn) # (b, 1)
    return pred