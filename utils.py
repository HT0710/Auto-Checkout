from typing import List, Tuple, Union
import yaml
import os


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
        except:
            raise TypeError(
                f"The 'value' parameter must be an int or tuple or list. Got {type(value)} instead."
            )
    if len(output) != max_dim:
        raise ValueError(
            f"The lenght of 'value' parameter must be equal to {max_dim}. Got {len(output)} instead."
        )
    return output
