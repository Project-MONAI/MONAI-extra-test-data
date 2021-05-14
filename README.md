# monai-test-data [![CI Build](https://github.com/Project-MONAI/MONAI-extra-test-data/workflows/build/badge.svg?branch=main)](https://github.com/Project-MONAI/MONAI-extra-test-data/commits/main)

Repository for storing networks and their forward data. 

The data in this repository is used by the main [MONAI repository](https://github.com/Project-MONAI/MONAI) as a unit test. This repository is checked out, the environmantal variable is `MONAI_EXTRA_TEST_DATA` is set and then the unit test can be run.

## Running the tests

```bash
git clone https://github.com/Project-MONAI/MONAI-extra-test-data.git --depth 1
git clone https://github.com/Project-MONAI/MONAI.git --depth 1

export MONAI_EXTRA_TEST_DATA=MONAI-extra-test-data

cd MONAI
pip install --user --upgrade -r requirements-min.txt
python -m unittest -v tests/test_network_consistency.py

```

## How to create a new test

```python
# model name
model_name = "UNet"
# network params
params = {
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
input_shape = (1, params["in_channels"], 64, 64, 64)
# create data
create_model_test_data(model_name, params, input_shape)
```
