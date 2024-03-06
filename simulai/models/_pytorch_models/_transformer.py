import copy
from typing import Tuple, Union

import numpy as np
import torch

from simulai.activations import TrainableActivation
from simulai.regression import DenseNetwork, Linear
from simulai.templates import NetworkTemplate, as_tensor, guarantee_device


class BaseTemplate(NetworkTemplate):
    def __init__(self, device: str = "cpu"):
        """Template used for sharing fundamental methods with the
        children transformer-like encoders and decoders.

        """

        super(BaseTemplate, self).__init__()
        self.device = device

    def _activation_getter(
        self, activation: Union[str, torch.nn.Module]
    ) -> torch.nn.Module:
        """It configures the activation functions for the transformer layers.

        Args:
            activation (Union[str, torch.nn.Module]): Activation function to be used in all the network layers

        Raises:
            Exception: When the activation function is not supported.


        """

        if isinstance(activation, torch.nn.Module):
            return encoder_activation
        elif isinstance(activation, str):
            act = self._get_operation(operation=activation, is_activation=True)

            if isinstance(act, TrainableActivation):
                act.setup(device=self.device)

            return act
        else:
            raise Exception(f"The activation {activation} is not supported.")


class BasicEncoder(BaseTemplate):
    def __init__(
        self,
        num_heads: int = 1,
        activation: Union[str, torch.nn.Module] = "relu",
        mlp_layer: torch.nn.Module = None,
        embed_dim: Union[int, Tuple] = None,
        device: str = "cpu",
    ) -> None:
        """Generic transformer encoder.

        Args:
            num_heads (int, optional): Number of attention heads for the self-attention layers. (Default value = 1)
            activation (Union[str, torch.nn.Module], optional): Activation function to be used in all the network layers (Default value = 'relu')
            mlp_layer (torch.nn.Module, optional): A Module object representing the MLP (Dense) operation. (Default value = None)
            embed_dim (Union[int, Tuple], optional): Dimension used for the transfoirmer embedding. (Default value = None)

        """

        super(BasicEncoder, self).__init__(device=device)

        self.num_heads = num_heads

        self.embed_dim = embed_dim

        self.activation_1 = self._activation_getter(activation=activation)
        self.activation_2 = self._activation_getter(activation=activation)

        self.mlp_layer = mlp_layer

        self.self_attention = torch.nn.MultiheadAttention(
            embed_dim=self.embed_dim, num_heads=self.num_heads, batch_first=True
        )

        self.add_module("mlp_layer", self.mlp_layer)
        self.add_module("self_attention", self.self_attention)

        # Attention heads are not being included in the weights regularization
        self.weights = list()
        self.weights += self.mlp_layer.weights

    def forward(
        self, input_data: Union[torch.Tensor, np.ndarray] = None
    ) -> torch.Tensor:
        """

        Args:
            input_data (Union[torch.Tensor, np.ndarray], optional): The input dataset. (Default value = None)

        Returns:
            torch.Tensor: The output generated by the encoder.

        """

        h = input_data
        h1 = self.activation_1(h)
        h = h + self.self_attention(h1, h1, h1)[0]
        h2 = self.activation_2(h)
        h = h + self.mlp_layer(h2)

        return h


class BasicDecoder(BaseTemplate):
    def __init__(
        self,
        num_heads: int = 1,
        activation: Union[str, torch.nn.Module] = "relu",
        mlp_layer: torch.nn.Module = None,
        embed_dim: Union[int, Tuple] = None,
        device: str = "cpu",
    ):
        """Generic transformer decoder.

        Args:
            num_heads (int, optional): Number of attention heads for the self-attention layers. (Default value = 1)
            activation (Union[str, torch.nn.Module], optional): Activation function to be used in all the network layers (Default value = 'relu')
            mlp_layer (torch.nn.Module, optional): A Module object representing the MLP (Dense) operation. (Default value = None)
            embed_dim (Union[int, Tuple], optional): Dimension used for the transfoirmer embedding. (Default value = None)

        """

        super(BasicDecoder, self).__init__()

        self.num_heads = num_heads

        self.embed_dim = embed_dim

        self.activation_1 = self._activation_getter(activation=activation)
        self.activation_2 = self._activation_getter(activation=activation)

        self.mlp_layer = mlp_layer

        self.self_attention = torch.nn.MultiheadAttention(
            embed_dim=self.embed_dim, num_heads=self.num_heads, batch_first=True
        )
        self.add_module("mlp_layer", self.mlp_layer)
        self.add_module("self_attention", self.self_attention)

        # Attention heads are not being included in the weights regularization
        self.weights = list()
        self.weights += self.mlp_layer.weights

    def forward(
        self,
        input_data: Union[torch.Tensor, np.ndarray] = None,
        encoder_output: torch.Tensor = None,
    ) -> torch.Tensor:
        """

        Args:
            input_data (Union[torch.Tensor, np.ndarray], optional): The input dataset (in principle, the same input used for the encoder). (Default value = None)
            encoder_output (torch.Tensor, optional): The output provided by the encoder stage. (Default value = None)

        Returns:
            torch.Tensor: The decoder output.

        """

        h = input_data
        h1 = self.activation_1(h)
        h = h + self.self_attention(h1, encoder_output, encoder_output)[0]
        h2 = self.activation_2(h)
        h = h + self.mlp_layer(h2)

        return h


class Transformer(NetworkTemplate):
    def __init__(
        self,
        num_heads_encoder: int = 1,
        num_heads_decoder: int = 1,
        embed_dim_encoder: Union[int, Tuple] = None,
        embed_dim_decoder: Union[int, Tuple] = None,
        output_dim: Union[int, Tuple] = None,
        encoder_activation: Union[str, torch.nn.Module] = "relu",
        decoder_activation: Union[str, torch.nn.Module] = "relu",
        encoder_mlp_layer_config: dict = None,
        decoder_mlp_layer_config: dict = None,
        number_of_encoders: int = 1,
        number_of_decoders: int = 1,
        devices: Union[str, list] = "cpu",
    ) -> None:
        r"""A classical encoder-decoder transformer:

        Graphical example:

        Example::

             U -> ( Encoder_1 -> Encoder_2 -> ... -> Encoder_N ) -> u_e

            (u_e, U) -> ( Decoder_1 -> Decoder_2 -> ... Decoder_N ) -> V

        Args:
            num_heads_encoder (int, optional): The number of heads for the self-attention layer of the encoder. (Default value = 1)
            num_heads_decoder (int, optional): The number of heads for the self-attention layer of the decoder. (Default value = 1)
            embed_dim_encoder (int, optional): The dimension of the embedding for the encoder. (Default value = Union[int, Tuple])
            embed_dim_decoder (int, optional): The dimension of the embedding for the decoder. (Default value = Union[int, Tuple])
            output_dim (int, optional): The dimension of the final output. (Default value = Union[int, Tuple])
            encoder_activation (Union[str, torch.nn.Module], optional): The activation to be used in all the encoder layers. (Default value = 'relu')
            decoder_activation (Union[str, torch.nn.Module], optional): The activation to be used in all the decoder layers. (Default value = 'relu')
            encoder_mlp_layer_config (dict, optional): A configuration dictionary to instantiate the encoder MLP layer.weights (Default value = None)
            decoder_mlp_layer_config (dict, optional): A configuration dictionary to instantiate the encoder MLP layer.weights (Default value = None)
            number_of_encoders (int, optional): The number of encoders to be used. (Default value = 1)
            number_of_decoders (int, optional): The number of decoders to be used. (Default value = 1)

        """

        super(Transformer, self).__init__()

        self.num_heads_encoder = num_heads_encoder
        self.num_heads_decoder = num_heads_decoder

        self.embed_dim_encoder = embed_dim_encoder
        self.embed_dim_decoder = embed_dim_decoder

        if not output_dim:
            self.output_dim = embed_dim_decoder
        else:
            self.output_dim = output_dim

        self.encoder_mlp_layer_dict = encoder_mlp_layer_config
        self.decoder_mlp_layer_dict = decoder_mlp_layer_config

        self.number_of_encoders = number_of_encoders
        self.number_of_decoders = number_of_encoders

        self.encoder_activation = encoder_activation
        self.decoder_activation = decoder_activation

        self.encoder_mlp_layers_list = list()
        self.decoder_mlp_layers_list = list()

        # Determining the kind of device in which the modelwill be executed
        self.device = self._set_device(devices=devices)

        # Creating independent copies for the MLP layers which will be used
        # by the multiple encoders/decoders.
        for e in range(self.number_of_encoders):
            self.encoder_mlp_layers_list.append(
                DenseNetwork(**self.encoder_mlp_layer_dict)
            )

        for d in range(self.number_of_decoders):
            self.decoder_mlp_layers_list.append(
                DenseNetwork(**self.decoder_mlp_layer_dict)
            )

        # Defining the encoder architecture
        self.EncoderStage = torch.nn.Sequential(
            *[
                BasicEncoder(
                    num_heads=self.num_heads_encoder,
                    activation=self.encoder_activation,
                    mlp_layer=self.encoder_mlp_layers_list[e],
                    embed_dim=self.embed_dim_encoder,
                    device=self.device,
                )
                for e in range(self.number_of_encoders)
            ]
        )

        # Defining the decoder architecture
        self.DecoderStage = torch.nn.ModuleList(
            [
                BasicDecoder(
                    num_heads=self.num_heads_decoder,
                    activation=self.decoder_activation,
                    mlp_layer=self.decoder_mlp_layers_list[d],
                    embed_dim=self.embed_dim_decoder,
                    device=self.device,
                )
                for d in range(self.number_of_decoders)
            ]
        )

        self.weights = list()

        for e, encoder_e in enumerate(self.EncoderStage):
            self.weights += encoder_e.weights
            self.add_module(f"encoder_{e}", encoder_e)

        for d, decoder_d in enumerate(self.DecoderStage):
            self.weights += decoder_d.weights
            self.add_module(f"decoder_{d}", decoder_d)

        self.final_layer = Linear(
            input_size=self.embed_dim_decoder, output_size=self.output_dim
        )
        self.add_module("final_linear_layer", self.final_layer)

        #  Sending everything to the proper device
        self.EncoderStage = self.EncoderStage.to(self.device)
        self.DecoderStage = self.DecoderStage.to(self.device)
        self.final_layer = self.final_layer.to(self.device)

    @as_tensor
    def forward(
        self, input_data: Union[torch.Tensor, np.ndarray] = None
    ) -> torch.Tensor:
        """

        Args:
            input_data (Union[torch.Tensor, np.ndarray], optional): The input dataset. (Default value = None)

        Returns:
            torch.Tensor: The transformer output.

        """

        encoder_output = self.EncoderStage(input_data)

        current_input = input_data
        for decoder in self.DecoderStage:
            output = decoder(input_data=current_input, encoder_output=encoder_output)
            current_input = output

        # Final linear operation
        final_output = self.final_layer(output)

        return final_output

    def summary(self):
        """It prints a general view of the architecture."""

        print(self)
