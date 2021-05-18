from create_data import create_model_test_data

# network params
params = {
    "dimensions": 3,
    "in_shape": (1, 64, 128, 128),
    "out_channels": 1,
    "latent_size": 2,
    "channels": (4, 8, 16),
    "strides": (2, 2, 2),
    "num_res_units": 0,
}
# in shape
input_shape = (1,) + params["in_shape"]
# create data
create_model_test_data("VarAutoEncoder", params, input_shape)
