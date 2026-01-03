
import numpy as np
import numbers

def check_type(arg, expected_type, arg_name):
    if not isinstance(arg, expected_type):
        raise TypeError(f"Expected {arg_name} to be of type {expected_type}, but got {type(arg).__name__} instead.")

def check_scalar_value_bounds(arg, arg_name, min_value=float('-inf'), max_value=float('inf'), strictly_positive=False):
    check_type(arg, numbers.Number, arg_name)
    if arg < min_value:
        raise ValueError(f"Expected {arg_name} to be greater or equal than {min_value}, but got {arg_name}={arg} instead.")
    if arg > max_value: 
        raise ValueError(f"Expected {arg_name} to be less or equal than {max_value}, but got {arg_name}={arg} instead.")
    if strictly_positive and arg < 0.0:
        raise ValueError(f"Expected {arg_name} to be strictly positive, but got {arg_name}={arg} instead.")
    
def check_callable(arg, arg_name):
    if not callable(arg):
        raise TypeError(f"Expected {arg_name} to be a callable object, but got {type(arg).__name__} instead.")

def check_list_length(arg, expected_length, arg_name):
    if not len(arg) == expected_length:
        raise ValueError(f"Expected '{arg_name}' to be a list of length {expected_length}, but got length {len(arg)} instead.")
    
def check_array_dimensions(arg, expected_dim, arg_name):
    check_type(arg, np.ndarray, arg_name)
    if arg.ndim != expected_dim:
        raise ValueError(f"Expected {arg_name} to be a {expected_dim}-dimensional array, but got a {arg.ndim}-dimensional array instead.")
    
def check_array_shape(arg, expected_shape, arg_name):
    check_array_dimensions(arg, len(expected_shape), arg_name)
    if arg.shape != expected_shape:
        raise ValueError(f"Expected {arg_name} to be of shape {expected_shape}, but got shape {arg.shape} instead.")
    
def check_array_in_simplex(arg, n, m, arg_name, tolerance=1e-6):
    check_array_shape(arg, (n,), arg_name)
    sum = 0
    for i in range(n):
        check_scalar_value_bounds(arg[i], f"{arg_name}[{i}]", min_value=0, max_value=m)
        sum += arg[i]
    if sum > m + tolerance or sum < m - tolerance:
        raise ValueError(f"Expected sum({arg_name})={m}, but got sum({arg_name})={sum} instead (tolerance={tolerance}).") 

def check_valid_list(arg, length, internal_type, name, strictly_positive=False):
    check_type(arg, list, name)
    check_list_length(arg, length, name)
    if strictly_positive:
        cond = all(isinstance(i, internal_type) and i > 0 for i in arg)
        if not cond:
            raise ValueError(f'Input {name} is expected to be a list of {length} positive {internal_type}s. Got {arg} instead.')
    else:
        cond = all(isinstance(i, internal_type) for i in arg)
        if not cond:
            raise ValueError(f'Input {name} is expected to be a list of {length} {internal_type}s. Got {arg} instead.')
    
def check_function_signature(arg : object, expected_input_shapes : list[tuple], expected_output_shape : tuple, name : str):
    check_callable(arg, name)
    try:
        inputs = []
        for shape in expected_input_shapes:
            inputs.append(np.ones(shape=shape))
        out = arg(*inputs)
    except:
        raise ValueError(f'Provided {name} raised an error when called with an input of {len(expected_input_shapes)} np.ndarrays of shapes={expected_input_shapes}')
    
    try:
        check_type(out, np.ndarray, name + '(dummy_input)')
        check_array_shape(out, expected_output_shape, name + '(dummy_input)')
    except:
        raise ValueError(f'Provided {name} must return as output a np.ndarray of shape={expected_output_shape} when called with an input of {len(expected_input_shapes)} np.ndarrays of shapes={expected_input_shapes}')