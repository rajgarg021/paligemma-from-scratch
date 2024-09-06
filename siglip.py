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


class SigLIPMLP(nn.Module):
    """
    This class implements the Multi-Layer Perceptron (MLP) component of the SigLIP vision transformer.

    The MLP consists of two fully connected layers with a GELU activation function in between.
    It's used as part of the feed-forward network in each transformer encoder layer.

    The forward pass of this MLP can be summarized as:
    1. Project input from hidden_size to intermediate_size using fc1
    2. Apply GELU activation
    3. Project back from intermediate_size to hidden_size using fc2

    This structure allows the model to capture complex non-linear relationships in the data.
    """

    def __init__(self, config: SigLIPVisionConfig):
        super().__init__()
        self.config = config
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor):
        # (B, num_patches, embed_dim) -> (B, num_patches, intermediate_size)
        hidden_states = self.fc1(hidden_states)
        # (B, num_patches, intermediate_size)
        hidden_states = nn.functional.gelu(hidden_states, approximate="tanh")
        # (B, num_patches, intermediate_size) -> (B, num_patches, embed_dim)
        hidden_states = self.fc2(hidden_states)

        return hidden_states


class SigLIPAttention(nn.Module):
    """
    This class implements the Multi-Head Attention mechanism for the SigLIP vision transformer.

    The attention mechanism allows the model to focus on different parts of the input sequence
    when processing each element. It does this by:
    1. Projecting the input into query, key, and value representations
    2. Computing attention scores between query and key
    3. Using these scores to create a weighted sum of the values

    The "multi-head" aspect means this process is performed in parallel across multiple sets
    of projections, allowing the model to jointly attend to information from different
    representation subspaces.

    The forward pass of this attention mechanism can be summarized as:
    1. Project input to query, key, and value states
    2. Split these projections into multiple heads
    3. Compute scaled dot-product attention for each head
    4. Concatenate results from all heads
    5. Project the concatenated result to the output dimension

    This structure allows the model to capture complex relationships and dependencies
    in the input data across different positions and feature dimensions.
    """

    def __init__(self, config: SigLIPVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim ** -0.5 # 1 / sqrt(self.head_dim)
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, hidden_states: torch.Tensor):
        # (B, num_patches, embed_dim)
        batch_size, seq_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # (B, num_patches, embed_dim) -> (B, num_patches, num_heads, head_dim) -> (B, num_heads, num_patches, head_dim)
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # calculating the attention using the formula Q * K^T / sqrt(d_k)
        # attn_weights: (B, num_heads, num_patches, num_patches)
        attn_weights = (torch.matmul(query_states, key_states.transpose(2, 3)) * self.scale) 

        if attn_weights.size() != (batch_size, self.num_heads, seq_len, seq_len):
            raise ValueError(
                f"Attention weights should be of size {(batch_size, self.num_heads, seq_len, seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        # applying the softmax row-wise, attn_weights: [B, num_heads, num_patches, num_patches]
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        # applying dropout only during training
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        # multiplying the attention weights by the value states, attn_output: [B, num_heads, num_patches, head_dim]
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (batch_size, self.num_heads, seq_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(batch_size, self.num_heads, seq_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )
        
        # (B, num_heads, num_patches, head_dim) -> (B, num_patches, num_heads, head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous()
        # (B, num_patches, num_heads, head_dim) -> (B, num_patches, embed_dim)
        attn_output = attn_output.reshape(batch_size, seq_len, self.embed_dim)
        # (B, num_patches, embed_dim)
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights


class SigLIPEncoderLayer(nn.Module):
    """
    This class implements a single encoder layer of the SigLIP vision transformer.
    
    It consists of two main components:
    1. Multi-head self-attention mechanism (SigLIPAttention)
    2. Feed-forward neural network (SigLIPMLP)
    
    The encoder layer applies these components in sequence, with layer normalization
    and residual connections around each component. This structure allows the model
    to learn complex relationships within the input data while maintaining gradient flow.
    
    The forward pass of this layer can be summarized as:
    1. Apply layer normalization to the input
    2. Pass through self-attention
    3. Add residual connection
    4. Apply layer normalization
    5. Pass through MLP
    6. Add residual connection
    
    This architecture is based on the original Transformer model, adapted for vision tasks.
    """

    def __init__(self, config: SigLIPVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.self_attn = SigLIPAttention()
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = SigLIPMLP()
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor):
        # (B, num_patches, embed_dim)
        residual = hidden_states
        # (B, num_patches, embed_dim) -> (B, num_patches, embed_dim)
        hidden_states = self.layer_norm1(hidden_states)
        # (B, num_patches, embed_dim) -> (B, num_patches, embed_dim) 
        hidden_states, _ = self.self_attn(hidden_states=hidden_states)
        # skip connection
        hidden_states = residual + hidden_states

        residual = hidden_states
        # (B, num_patches, embed_dim) -> (B, num_patches, embed_dim)
        hidden_states = self.layer_norm2(hidden_states)
        # (B, num_patches, embed_dim) -> (B, num_patches, embed_dim)
        hidden_states = self.mlp(hidden_states)
        # skip connection
        hidden_states = residual + hidden_states

        return hidden_states


class SigLIPEncoder(nn.Module):
    """
    The SigLIPEncoder class represents the main encoder component of the SigLIP vision model.
    
    It consists of a stack of SigLIPEncoderLayers, where the number of layers is determined
    by the configuration. This class is responsible for processing the input embeddings
    through multiple transformer-like encoder layers.

    Key features:
    1. Utilizes a variable number of SigLIPEncoderLayers based on the configuration.
    2. Processes input embeddings sequentially through all encoder layers.
    3. Maintains the input shape throughout the encoding process.

    The forward pass of this encoder can be summarized as:
    1. Take input embeddings (typically from SigLIPVisionEmbeddings)
    2. Pass the embeddings through each encoder layer sequentially
    3. Return the final encoded representation

    This architecture allows the model to learn hierarchical features and complex
    relationships within the input data, which is crucial for vision tasks.
    """

    def __init__(self, config:SigLIPVisionConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [SigLIPEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(self, input_embeds: torch.Tensor):
        # input_embeds: (B, num_patches, embed_dim)
        hidden_states = input_embeds

        for encoder_layer in self.layers:
            hidden_states = encoder_layer(hidden_states)
        
        # (B, num_patches, embed_dim)
        return hidden_states


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
