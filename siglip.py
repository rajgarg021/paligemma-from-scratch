import torch
import torch.nn as nn
from typing import Optional, Tuple


class SigLIPVisionConfig:
    """
    Defines the configuration for the vision component of the SigLIP model.

    This class defines the hyperparameters and architectural settings for the
    vision transformer used in SigLIP. It includes parameters such as the size
    of hidden layers, number of attention heads, image dimensions, and various
    other settings that determine the structure and behavior of the vision model.

    The class allows for easy configuration and modification of these parameters,
    facilitating experimentation with different model architectures.
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


class SigLIPVisionEmbeddings(nn.Module):
    """
    This class implements the embedding layer for the vision component of the SigLIP model.
    
    It performs the following operations:
    1. Converts input images into patches using a convolutional layer.
    2. Flattens and embeds these patches into a lower-dimensional space.
    3. Adds positional embeddings to provide spatial information to the model.
    
    The resulting embeddings serve as the input to the subsequent transformer layers
    in the SigLIP vision model.
    """

    def __init__(self, config: SigLIPVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.num_patches = (self.image_size // self.patch_size) ** 2
        
        self.patch_embeddings = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid", # valid is the same as no padding
        )

        self.num_positions = self.num_patches
        self.position_embeddings = nn.Embedding(num_embeddings=self.num_positions, embedding_dim=self.embed_dim)
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_positions).expand((1, -1)),
            persistent=False,
        )

    def forward(self, pixel_values: torch.Tensor):

        # convert image to patches
        # (B, C, H, W) -> (B, embed_dim, num_patches_h, num_patches_w)
        # where num_patches_h = image_height // patch_size and num_patches_w = image_width // patch_size
        patch_embeddings = self.patch_embeddings(pixel_values)

        # (B, embed_dim, num_patches_h, num_patches_w) -> (B, embed_dim, num_patches) -> (B, num_patches, embed_dim)
        embeddings = patch_embeddings.flatten(2).transpose(1, 2)

        # adding position embeddings
        position_embeddings = self.position_embeddings(self.position_ids)
        embeddings = embeddings + position_embeddings
        
        # (B, num_patches, embed_dim)
        return embeddings

        
class SigLIPVisionTransformer(nn.Module):
    """
    This class implements the Vision Transformer (ViT) architecture for the SigLIP model.
    
    It processes image inputs by:
    1. Embedding the image patches
    2. Passing the embedded patches through a series of transformer encoder layers
    3. Applying a final layer normalization
    
    The class takes a SigLIPVisionConfig object to define its architecture and hyperparameters.
    
    Key components:
    - embeddings: Converts image patches to embeddings
    - encoder: Processes embeddings through transformer layers
    - layernorm: Applies final normalization to the output
    
    The forward method takes pixel values as input and returns the processed
    image representations.
    """

    def __init__(self, config: SigLIPVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = SigLIPVisionEmbeddings(config)
        self.encoder = SigLIPEncoder(config)
        self.layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    def forward(self, pixel_values: torch.Tensor):
        # pixel_values: (B, C, H, W)
        embeddings_output = self.embeddings(pixel_values)
        encoder_output = self.encoder(input_embeds=embeddings_output)
        output = self.layernorm(encoder_output)

        # output: (B, num_patches, embed_dim)
        return output
    
    
class SigLIPVisionModel(nn.Module):
    """
    This class is a wrapper around the SigLIPVisionTransformer.
    It takes pixel values as input and processes them through the vision transformer.

    This class serves as the main interface for the vision component of the SigLIP model.
    It encapsulates the configuration and the actual transformer model, providing a
    simple forward method that can be easily integrated into the larger SigLIP architecture.

    Attributes:
        config (SigLIPVisionConfig): Configuration object for the vision model.
        vision_model (SigLIPVisionTransformer): The underlying vision transformer model.

    Methods:
        forward(pixel_values): Processes the input pixel values through the vision transformer.
    """

    def __init__(self, config: SigLIPVisionConfig):
        super().__init__()
        self.config = config
        self.vision_model = SigLIPVisionTransformer(config)

    def forward(self, pixel_values) -> Tuple:
        # (B, C, H, W) -> (B, num_patches, embed_dim)
        return self.vision_model(pixel_values=pixel_values)
