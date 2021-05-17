from create_data import create_model_test_data

# network params
params = {
    "spatial_dims": 3,
    "upsample_mode": "nearest",
    "out_channels": 2,
    "psp_block_num": 3
}
# in shape
input_shape = (1, 1, 96, 128, 32)
# create data
create_model_test_data("AHNet", params, input_shape)
