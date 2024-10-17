import numpy as np


def inner_product_forward(input, layer, param):
    """
    Forward pass of inner product layer.

    Parameters:
    - input (dict): Contains the input data.
    - layer (dict): Contains the configuration for the inner product layer.
    - param (dict): Contains the weights and biases for the inner product layer.
    """
    # print(f"Inner Product Layer - Input shape: {input['data'].shape}")

    d, k = input["data"].shape
    # print(f"Weight shape: {param['w'].shape}")
    # print(f"Bias shape: {param['b'].shape}")

    n = param["w"].shape[1]

    ###### Fill in the code here ######

    # output_data = np.dot(W.T, input["data"]) + b.reshape(-1, 1)

    # output_data = input["data"].T.dot(param["w"]) + np.tile(param["b"], (k, 1))
    # output_data = input["data"].T.dot(param["w"]) + param["b"]
    output_data = np.dot(param["w"].T, input["data"])
    output_data += param["b"].reshape(-1, 1)




    # print(f"Output data shape: {output_data.shape}")

    # Initialize output data structure
    output = {
        "height": n,
        "width": 1,
        "channel": 1,
        "batch_size": k,
        "data": output_data # replace 'data' value with your implementation
    }
    # print(f"Output data sample (first 5 elements): {output_data.flatten()[:5]}")
   
    return output


def inner_product_backward(output, input_data, layer, param):
    """
    Backward pass of inner product layer.

    Parameters:
    - output (dict): Contains the output data.
    - input_data (dict): Contains the input data.
    - layer (dict): Contains the configuration for the inner product layer.
    - param (dict): Contains the weights and biases for the inner product layer.
    """
    param_grad = {}
    ###### Fill in the code here ######
    # Replace the following lines with your implementation.
    # param_grad['b'] = np.sum(output['diff'], axis=1, keepdims=True)
    # param_grad['w'] = np.dot(input_data['data'], output['diff'].T)


    param_grad['w'] = np.dot(input_data['data'], output['diff'].T)
    param_grad['b'] = np.sum(output['diff'], axis=1)#, keepdims=True


    input_od = np.dot(param['w'], output['diff'])

    return param_grad, input_od