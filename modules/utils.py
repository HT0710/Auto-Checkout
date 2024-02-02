from typing import Union, Tuple, List
import yaml
import os

import torch


__all__ = ["device_handler", "workers_handler", "tuple_handler", "load_config"]


def device_handler(value: str = "auto") -> str:
    """
    Handles the specification of device choice.

    Args:
        value (str): The device specification. Valid options: ["auto", "cpu", "cuda"]. Default to "auto".

    Returns:
        str: The selected device string.

    Example:
        >>> device_handler("auto")
        'cuda'  # Returns 'cuda' if GPU is available, otherwise 'cpu'
    """

    # Check type
    if not isinstance(value, str):
        raise TypeError(
            f"The 'value' parameter must be a string. Got {type(value)} instead."
        )

    # Prepare
    value = value.strip().lower()

    # Check value
    if not (value in ["auto", "cpu", "gpu"] or value.startswith("cuda")):
        raise ValueError(
            f'Device options: ["auto", "cpu", "cuda"]. Got {value} instead.'
        )

    # Case 'auto'
    if value == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Case 'cpu'
    elif value == "cpu":
        device = "cpu"

    # Case 'gpu' or 'cuda'
    elif value == "gpu" or value.startswith("cuda"):
        if not torch.cuda.is_available():
            raise ValueError("CUDA device not found.")
        device = "cuda"

    return device


def workers_handler(value: Union[int, float]) -> int:
    """
    Calculate the number of workers based on an input value.

    Args:
        value (int | float): The input value to determine the number of workers.
            Int for a specific numbers. \
            Float for a specific portion. \
            Set to 0 to use all available cores.

    Returns:
        int: The computed number of workers for parallel processing.
    """

    # Get max workers
    max_workers = os.cpu_count()

    # If value is int
    if isinstance(value, int):
        if value < 0:
            workers = 1
        elif value == 0 or value > max_workers:
            workers = max_workers
        else:
            workers = value

    # If value is float
    elif isinstance(value, float):
        if value < 0:
            workers = 1
        elif value > 1:
            workers = max_workers
        else:
            workers = int(max_workers * value)

    # Wrong type
    else:
        raise TypeError(
            f"Workers value must be Int or Float. Got {type(workers)} instead."
        )

    return workers


def tuple_handler(value: Union[int, List[int], Tuple[int]], max_dim: int) -> Tuple:
    """
    Create a tuple with specified dimensions and values.

    Args:
        value (Union[int, List[int], Tuple[int]]): The value(s) to populate the tuple with.
            - If an integer is provided, a tuple with 'max_dim' elements, each set to this integer, is created.
            - If a tuple or list of integers is provided, it should have 'max_dim' elements.
        max_dim (int): The desired dimension (length) of the resulting tuple.

    Returns:
        Tuple: A tuple containing the specified values.

    Raises:
        TypeError: If 'max_dim' is not an integer or is less than or equal to 1.
        TypeError: If 'value' is not an integer, tuple, or list.
        ValueError: If the length of 'value' is not equal to 'max_dim'.
    """

    # Check max_dim
    if not isinstance(max_dim, int) and max_dim > 1:
        raise TypeError(
            f"The 'max_dim' parameter must be an int. Got {type(max_dim)} instead."
        )
    # Check value
    if isinstance(value, int):
        output = tuple([value] * max_dim)
    else:
        try:
            output = tuple(value)
        except TypeError:
            raise TypeError(
                f"The 'value' parameter must be an int or tuple or list. Got {type(value)} instead."
            )
    if len(output) != max_dim:
        raise ValueError(
            f"The lenght of 'value' parameter must be equal to {max_dim}. Got {len(output)} instead."
        )
    return output


def load_config(file_path: str):
    """
    Loads a YAML configuration file.

    Args:
        file_path (str): Path to the YAML configuration file.

    Returns:
        dict: Configuration parameters.
    """
    # Check if the file exists
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Config file not found: {file_path}")

    try:
        # Attempt to load the YAML file
        with open(file_path, "r") as file:
            config = yaml.safe_load(file)
        return config
    except yaml.YAMLError as e:
        # Handle YAML parsing errors
        raise ValueError(f"Error parsing YAML in {file_path}: {e}")
    except Exception as e:
        # Handle other unexpected errors
        raise RuntimeError(f"Unexpected error while loading config: {e}")
