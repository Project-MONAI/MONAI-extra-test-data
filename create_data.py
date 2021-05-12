import json
from typing import Any, Callable, Dict, Tuple

import torch


def create_model_test_data(
    model_class: Callable,
    model_params: Dict[str, Any],
    input_shape: Tuple[int],
    out_path_no_ext: str,
) -> None:
    """
    Create test data to check model consistency

    Args:
        model_class: Class of model to be tested.
        model_params: Dictionary of parameters to construct object.
        input_shape: Tuple of dimensions (B, C, H, W, [D]).
        out_path_no_ext: Path for saved objects (no extension as both torch.save and json.dump will be used).

    .. code-block:: python

        # model parameters
        unet_params = {
            "dimensions" : 3,
            "in_channels" : 4,
            "out_channels" : 2,
            "channels": (4, 8, 16, 32),
            "strides": (2, 4, 1),
            "kernel_size" : 5,
            "up_kernel_size" : 3,
            "num_res_units": 2,
            "act": "relu",
            "dropout": 0.1,
        }
        # in shape
        input_shape = (1, unet_params["in_channels"], 64, 64, 64)
        # create data
        create_model_test_data(UNet, unet_params, input_shape, "unet_test")
    """

    with open(out_path_no_ext + ".json", "w+") as f:
        json.dump(model_params, f)

    # Create model
    model = model_class(**model_params)
    model.eval()

    # Create input data
    num_elements = int(torch.Tensor(input_shape).prod())
    in_data = torch.arange(num_elements).reshape(input_shape).float()

    # Forward pass data
    out_data = model(in_data)

    # Save in data, out data and model
    to_save = {"in_data": in_data, "out_data": out_data, "model": model.state_dict()}
    torch.save(to_save, out_path_no_ext + ".pt")
