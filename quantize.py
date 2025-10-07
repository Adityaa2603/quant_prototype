import torch


def quantize(
    self,
    weights: torch.Tensor,
    granularity: Granularity = Granularity.PER_ROW,
    mapping_type: MappingType = MappingType.SYMMETRIC,
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

    if mapping_type == MappingType.SYMMETRIC:
        qmin, qmax = -128, 127
    elif mapping_type == MappingType.ASYMMETRIC:
        qmin, qmax = 0, 255
    else:
        raise ValueError(f"Unsupported mapping type: {mapping_type}")

    scales_list, zero_points_list = get_qparams(
        weights, qmin, qmax, mapping_type, granularity
    )

    if granularity == Granularity.PER_ROW:
        scales_tensor = torch.tensor(scales_list, dtype=torch.float32).unsqueeze(1)
        zero_points_tensor = torch.tensor(
            zero_points_list, dtype=torch.float32
        ).unsqueeze(1)

    elif granularity == Granularity.PER_TENSOR:
        scales_tensor = torch.tensor(scales_list[0], dtype=torch.float32)
        zero_points_tensor = torch.tensor(zero_points_list[0], dtype=torch.float32)

    else:
        raise ValueError(f"Unsupported granularity: {granularity}")

    if mapping_type == MappingType.SYMMETRIC:
        int8_weights = (
            torch.round(weights / scales_tensor).clamp(qmin, qmax).to(torch.int8)
        )
    elif mapping_type == MappingType.ASYMMETRIC:
        int8_weights = (
            torch.round((weights / scales_tensor) + zero_points_tensor)
            .clamp(qmin, qmax)
            .to(torch.uint8)
        )

    self.int8_weights = int8_weights
    self.scales = scales_tensor
    self.zero_points = zero_points_tensor
