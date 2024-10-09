import numpy as np
from gymnasium import spaces


def get_dtype_bounds(dtype: np.dtype):
    if np.issubdtype(dtype, np.floating):
        info = np.finfo(dtype)
        return info.min, info.max
    elif np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return info.min, info.max
    elif np.issubdtype(dtype, bool):
        return 0, 1
    else:
        raise TypeError(dtype)


def convert_observation_to_space(observation, prefix=""):
    if isinstance(observation, (dict)):
        space = spaces.Dict({k: convert_observation_to_space(v, prefix + "/" + k) for k, v in observation.items()})
    elif isinstance(observation, np.ndarray):
        shape = observation.shape
        dtype = observation.dtype
        dtype_min, dtype_max = get_dtype_bounds(dtype)
        low = np.full(shape, dtype_min)
        high = np.full(shape, dtype_max)
        space = spaces.Box(low, high, dtype=dtype)
    elif isinstance(observation, float):
        space = spaces.Box(-np.inf, np.inf, shape=[1], dtype=np.float32)
    elif isinstance(observation, np.float32):
        space = spaces.Box(-np.inf, np.inf, shape=[1], dtype=np.float32)
    else:
        raise NotImplementedError(type(observation), observation)

    return space
