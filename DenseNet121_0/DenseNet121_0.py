from create_data import create_model_test_data

# network params
params = {
    "spatial_dims": 2,
    "in_channels": 2,
    "out_channels": 3,
    "pretrained": True,
    "progress": True,
}
# in shape
input_shape = (1, params["in_channels"], 32, 64)
# create data
create_model_test_data("DenseNet121", params, input_shape)
