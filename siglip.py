import torch
import torch.nn as nn
from typing import Optional, Tuple


class SigLIPVisionConfig:
    """
    Defines the configuration for the vision component of the SigLIP model 
    """

    def __init__(
        self,
        hidden_size: int = 768,  # dimensionality of the encoder layers and the pooler layer
        num_hidden_layers: int = 12,  # number of hidden layers in the transformer encoder
        num_attention_heads: int = 12,  # number of attention heads for each attention layer
        intermediate_size: int = 3072,  # dimensionality of the "intermediate" (feed-forward) layer
        attention_dropout: float = 0.0,  # the dropout ratio for the attention probabilities
        layer_norm_eps: float = 1e-6,  # the epsilon used by the layer normalization layers
        image_size: int = 224,  # the size of input images
        patch_size: int = 16,  # the size of each image patch
        num_channels: int = 3,  # the number of channels in the input images

        # max_position_embeddings: int = 197,  # the maximum sequence length that this model might ever be used with
        # initializer_range: float = 0.02,  # the standard deviation of the truncated_normal_initializer for initializing all weight matrices
        # hidden_act: str = "gelu",  # the non-linear activation function in the encoder and pooler
        # hidden_dropout_prob: float = 0.0,  # the dropout probability for all fully connected layers
        # qkv_bias: bool = True,  # whether to add a bias to the query, key, and value projections

        **kwargs
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels

        # self.max_position_embeddings = max_position_embeddings
        # self.initializer_range = initializer_range
        # self.hidden_act = hidden_act
        # self.hidden_dropout_prob = hidden_dropout_prob
        # self.qkv_bias = qkv_bias
