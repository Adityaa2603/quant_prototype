from enums import MappingEnum, GranularityEnum
from get_qparams import get_qparams
import torch
import torch.nn.functional as F


def int8_forward(
    weight: torch.Tensor,
    input: torch.Tensor,
    scales: torch.Tensor,
    zero_points: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    # Cast int8/uint8 weights to input's dtype for computation
    casted_weights = weight.to(input.dtype)

    # Handle asymmetric zero_point if provided
    if zero_points is not None:
        casted_weights = casted_weights - zero_points

    # Linear transformation followed by scaling
    output = F.linear(input, casted_weights) * scales

    if bias is not None:
        output = output + bias

    return output


class QuantizedLinear(torch.nn.Module):
    def __init__(
        self,
        input_features: int,
        output_features: int,
        bias: bool = True,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.bias = bias
        self.dtype = dtype

        # Separate buffers for symmetric (int8) and asymmetric (uint8) weights
        self.register_buffer(
            "int8_weights",
            torch.randint(-128, 127, (output_features, input_features), dtype=torch.int8),
        )
        self.register_buffer(
            "uint8_weights",
            torch.zeros((output_features, input_features), dtype=torch.uint8),
        )

        self.register_buffer("scales", torch.randn((output_features), dtype=dtype))
        self.register_buffer("zero_points", torch.zeros((output_features), dtype=dtype))

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
            zero_points_tensor = torch.tensor(zero_points_list, dtype=torch.float32).unsqueeze(1)
        elif granularity == GranularityEnum.PER_TENSOR:
            scales_tensor = torch.tensor(scales_list[0], dtype=torch.float32)
            zero_points_tensor = torch.tensor(zero_points_list[0], dtype=torch.float32)
        else:
            raise ValueError(f"Unsupported granularity: {granularity}")

        if mapping_type == MappingEnum.SYMMETRIC:
            int_weights = (
                torch.round(weights / scales_tensor).clamp(qmin, qmax).to(torch.int8)
            )
            self.int8_weights.copy_(int_weights)
        elif mapping_type == MappingEnum.ASYMMETRIC:
            uint_weights = (
                torch.round((weights / scales_tensor) + zero_points_tensor)
                .clamp(qmin, qmax)
                .to(torch.uint8)
            )
            self.uint8_weights.copy_(uint_weights)

        self.scales.copy_(scales_tensor)
        self.zero_points.copy_(zero_points_tensor)

    def forward(self, input):
        weight = self.int8_weights if self.int8_weights.sum() != 0 else self.uint8_weights
        return int8_forward(weight, input, self.scales, self.zero_points, self.bias)