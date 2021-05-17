import json
import os
import torch
from typing import Any, Dict, Sequence

import monai.networks.nets as nets


def create_model_test_data(
    model_name: str,
    model_params: Dict[str, Any],
    input_shape: Sequence[int],
) -> None:
    """
    Create test data to check model consistency

    Args:
        model_class: Name of model to be tested.
        model_params: Dictionary of parameters to construct object.
        input_shape: Tuple of dimensions (B, C, H, W, [D]).

    .. code-block:: python

        # network params
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
        create_model_test_data("UNet", unet_params, input_shape)
    """
    base_folder = os.path.dirname(os.path.abspath(__file__))

    # get next unused folder
    i=0
    while True:
        out_folder = os.path.join(base_folder, f"{model_name.lower()}_{i}")
        if not os.path.isdir(out_folder):
            print("\n\nCreating output folder: " + out_folder)
            os.mkdir(out_folder)
            break
        i += 1
    out_path_no_ext = os.path.join(out_folder, f"{model_name.lower()}_{i}")

    # Create model
    model = nets.__dict__[model_name](**model_params)
    model.eval()

    # Create input data
    num_elements = int(torch.Tensor(input_shape).prod())
    in_data = torch.arange(num_elements).reshape(input_shape).float()

    # Forward pass data
    out_data = model(in_data)

    # Save in data, out data and model
    data_path = out_path_no_ext + ".pt"
    to_save = {"in_data": in_data, "out_data": out_data, "model": model.state_dict()}
    print("Writing data output to .pt: " + data_path)
    torch.save(to_save, data_path)

    # Save parameters
    json_params = out_path_no_ext + ".json"
    with open(json_params, "w+") as f:
        print("Writing network parameters to .json: " + json_params)
        json.dump(model_params, f)



# default
if __name__ == "__main__":

    # network params
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
    create_model_test_data("UNet", unet_params, input_shape)
