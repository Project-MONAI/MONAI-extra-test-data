# monai-test-data
test repository for storing networks and their forward data to ensure consistency

## How to create a new test

```python
from monai.networks.nets import UNet
unet_params = {
    "dimensions" : 3,
    "in_channels" : 4,
    "out_channels" : 2,
    "channels":(4, 8, 16, 32),
    "strides":(2, 4, 1),
    "kernel_size" : 5,
    "up_kernel_size" : 3,
    "num_res_units":2,
    "act":"relu",
    "dropout":0.1,
}
    in_size = (1, unet_params["in_channels"], 64, 64, 64)
    create_model_test_data(UNet, unet_params, in_size, "unet_0")
```