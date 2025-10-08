import torch
from typing import List, Tuple
from enums import MappingEnum, GranularityEnum


def compute_qparams(
    x_min: float, x_max: float, qmin: int, qmax: int, mapping_type: MappingEnum
) -> Tuple[float, int]:
    """
    Compute quantization parameters for a given tensor range.

    Args:
        x_min: Minimum value of the tensor (float)
        x_max: Maximum value of the tensor (float)
        qmin: Minimum quantized integer (e.g., -128)
        qmax: Maximum quantized integer (e.g., 127)
        mapping_type: MappingEnum.SYMMETRIC or MappingEnum.ASYMMETRIC

    Returns:
        Tuple containing:
            scale (float): scaling factor
            zero_point (int): zero point
    """

    eps = 1e-8
    x_min, x_max = float(x_min), float(x_max)

    # degenerate case: all values identical -> avoid division by zero
    if abs(x_max - x_min) < eps:
        if mapping_type == MappingEnum.SYMMETRIC:
            return 1.0, 0
        else:  # asymmetric
            scale = 1.0
            zp = qmin - round(x_min / scale)
            zp = int(min(max(zp, qmin), qmax))
            return scale, zp

    if mapping_type == MappingEnum.SYMMETRIC:
        max_abs = max(abs(x_min), abs(x_max))
        qmax_abs = max(abs(qmin), abs(qmax))
        scale = max_abs / qmax_abs
        zero_point = 0

    elif mapping_type == MappingEnum.ASYMMETRIC:
        scale = (x_max - x_min) / (qmax - qmin)
        zero_point = qmin - round(x_min / scale)
        zero_point = int(min(max(zero_point, qmin), qmax))

    else:
        raise ValueError(f"Unsupported mapping type: {mapping_type}")

    return scale, zero_point


def get_qparams(
    weight_tensor: torch.Tensor,
    qmin: int,
    qmax: int,
    mapping_type: MappingEnum,
    granularity: GranularityEnum,
) -> Tuple[List[float], List[int]]:
    """
    Compute per-row or per-tensor quantization parameters for a weight tensor.

    Args:
        weight_tensor: Tensor containing weights to quantize, shape (out_features, in_features)
        qmin: Minimum quantized integer value (e.g., -128)
        qmax: Maximum quantized integer value (e.g., 127)
        mapping_type: MappingEnum.SYMMETRIC or MappingEnum.ASYMMETRIC
        granularity: GranularityEnum.PER_ROW or GranularityEnum.PER_TENSOR

    Returns:
        Tuple containing:
            scales (List[float]): scale factors for each row or tensor
            zero_points (List[int]): zero points for each row or tensor
    """

    scales = []
    zero_points = []

    if granularity == GranularityEnum.PER_ROW:
        x_min_vals = weight_tensor.min(dim=1).values
        x_max_vals = weight_tensor.max(dim=1).values
        for x_min_val, x_max_val in zip(x_min_vals, x_max_vals):
            scale, zero_point = compute_qparams(
                x_min_val.item(), x_max_val.item(), qmin, qmax, mapping_type
            )

            scales.append(scale)
            zero_points.append(zero_point)

    # elif granularity == Granularity.PER_COLUMN:
    #     x_min_vals = weight_tensor.min(dim=0).values
    #     x_max_vals = weight_tensor.max(dim=0).values
    #     for i in range(weight_tensor.shape[1]):
    #         scale, zero_point = compute_qparams(
    #             x_min_vals[i].item(),
    #             x_max_vals[i].item(),
    #             qmin, qmax,
    #             mapping_type
    #         )
    #         scales.append(scale)
    #         zero_points.append(zero_point)

    elif granularity == GranularityEnum.PER_TENSOR:
        x_min = weight_tensor.min().item()
        x_max = weight_tensor.max().item()
        scale, zero_point = compute_qparams(x_min, x_max, qmin, qmax, mapping_type)
        scales.append(scale)
        zero_points.append(zero_point)

    else:
        raise ValueError(f"Unsupported granularity: {granularity}")

    return scales, zero_points
