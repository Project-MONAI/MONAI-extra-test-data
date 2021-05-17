from create_data import create_model_test_data
strides = (2, 2, 2, 2)

# network params
params = {
    "spatial_dims": 3,
    "in_channels": 4,
    "out_channels": 2,
    "kernel_size": (3, 3, 3, 1),
    "strides": strides,
    "upsample_kernel_size": strides[1:],
}
# in shape
input_shape = (1, params["in_channels"], 64, 64, 64)
# create data
create_model_test_data("DynUNet", params, input_shape)
