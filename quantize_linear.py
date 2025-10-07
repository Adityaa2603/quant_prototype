from enums import MappingEnum, GranularityEnum
from get_qparams import get_qparams
import torch
import torch.nn.functional as F


def int8_forward(
    weight: torch.Tensor,
    input: torch.Tensor,
    scales: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Performs a forward pass for a quantized linear layer using int8 weights.

    Args:
        weight (torch.Tensor): Quantized weight tensor of shape (output_featuresures, input_featuresures), dtype int8.
        input (torch.Tensor): Input tensor of shape (batch_size, input_featuresures), typically float32 or float16.
        scales (torch.Tensor): Scale tensor of shape (output_featuresures,) used to dequantize the weights.
        bias (torch.Tensor | None, optional): Optional bias tensor of shape (1, output_featuresures). Defaults to None.

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, output_featuresures) after applying the linear transformation
                      and scaling. Same dtype as `input`.
    """
    # Cast int8 weights to input's dtype for computation
    casted_weights = weight.to(input.dtype)
    # Linear transformation followed by scaling
    output = F.linear(input, casted_weights) * scales

    if bias is not None:
        output = output + bias

    return output


class QuantizedLinear(torch.nn.Module):
    """
    A linear layer module with quantized int8 weights.

    Attributes:
        input_features (int): Number of input features.
        output_features (int): Number of output features.
        bias (bool): Whether to include a bias term.
        dtype (torch.dtype): Data type for bias and computation.
        int8_weights (torch.Tensor): Stored int8 weight tensor.
        scales (torch.Tensor | None): Scale tensor for dequantizing weights.
        zero_points (torch.Tensor | None): Zero-point tensor (for asymmetric quantization, currently unused).
        bias (torch.Tensor | None): Bias tensor if bias=True, else None.
    """

    def __init__(
        self,
        input_features: int,
        output_features: int,
        bias: bool = True,
        dtype: torch.dtype = torch.float32,
    ):
        """
        Initializes a QuantizedLinear layer.

        Args:
            input_features (int): Number of input features.
            output_features (int): Number of output features.
            bias (bool, optional): If True, adds a learnable bias to the output. Defaults to True.
            dtype (torch.dtype, optional): Data type for bias and computation. Defaults to torch.float32.
        """
        super().__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.bias = bias
        self.dtype = dtype

        # Initialize int8 weights randomly
        self.register_buffer(
            "int8_weights",
            torch.randint(-128, 127, (output_features, input_features), dtype=torch.int8),
        )

        # Scale and zero-point buffers for quantization
        self.register_buffer("scales", torch.randn((output_features), dtype=dtype))
        self.register_buffer("zero_points", torch.randn((1, output_features), 
                                             dtype=dtype))

        # Optional bias
        if bias:
            self.register_buffer("bias", torch.randn((1, output_features), dtype=dtype))
        else:
            self.bias = None

    def quantize(
        self,
        weights: torch.Tensor,
        granularity: GranularityEnum = GranularityEnum.PER_ROW,
        mapping_type: MappingEnum = MappingEnum.SYMMETRIC,
    ):
        """
        Quantizes a given weight tensor into int8 format based on the specified granularity
        and mapping type. Computes scale and zero-point tensors, then performs quantization.

        Args:
            self: The object that stores quantized weights and parameters.
            weights (torch.Tensor): The input weight tensor to be quantized.
            granularity (Granularity, optional): Determines the quantization resolution.
                - Granularity.PER_TENSOR → Single scale/zero-point for entire tensor.
                - Granularity.PER_ROW → Separate scale/zero-point for each row.
                Default is Granularity.PER_ROW.
            mapping_type (MappingType, optional): Defines quantization mapping behavior.
                - MappingType.SYMMETRIC → Values are centered around zero ([-128, 127]).
                - MappingType.ASYMMETRIC → Values shifted to positive range ([0, 255]).
                Default is MappingType.SYMMETRIC.

        Returns:
            None.
            Stores the following attributes in `self`:
                - self.int8_weights (torch.Tensor): Quantized int8/uint8 weight tensor.
                - self.scales (torch.Tensor): Scale tensor used for quantization.
                - self.zero_points (torch.Tensor): Zero-point tensor used for quantization.

        Raises:
            ValueError: If `weights` is None, or if an unsupported granularity or mapping type is provided.

        """

        if weights is None:
            raise ValueError("Expected weights tensor for quantization.")

        if mapping_type == MappingEnum.SYMMETRIC:
            qmin, qmax = -128, 127
        elif mapping_type == MappingEnum.ASYMMETRIC:
            qmin, qmax = 0, 255
        else:
            raise ValueError(f"Unsupported mapping type: {mapping_type}")

        scales_list, zero_points_list = get_qparams(
            weights, qmin, qmax, mapping_type, granularity
        )

        if granularity == GranularityEnum.PER_ROW:
            scales_tensor = torch.tensor(scales_list, dtype=torch.float32).unsqueeze(1)
            zero_points_tensor = torch.tensor(
                zero_points_list, dtype=torch.float32
            ).unsqueeze(1)

        elif granularity == GranularityEnum.PER_TENSOR:
            scales_tensor = torch.tensor(scales_list[0], dtype=torch.float32)
            zero_points_tensor = torch.tensor(zero_points_list[0], dtype=torch.float32)

        else:
            raise ValueError(f"Unsupported granularity: {granularity}")

        if mapping_type == MappingEnum.SYMMETRIC:
            int8_weights = (
                torch.round(weights / scales_tensor).clamp(qmin, qmax).to(torch.int8)
            )
        elif mapping_type == MappingEnum.ASYMMETRIC:
            int8_weights = (
                torch.round((weights / scales_tensor) + zero_points_tensor)
                .clamp(qmin, qmax)
                .to(torch.uint8)
            )

        self.int8_weights.copy_(int8_weights)
        self.scales.copy_(scales_tensor)
        self.zero_points.copy_(zero_points_tensor)


    def forward(self, input):
        return int8_forward(self.int8_weights, input, self.scales, self.bias)
