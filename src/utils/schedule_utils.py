from typing import List, Tuple


def set_schedule(
    num_steps: int,
    T_min: float = 0.0001,
    T_max: float = 0.01,
) -> List[float]:
    """
    Create schedules for temperature and energy coefficient for optimization steps.

    Args:
        num_steps (int): Total number of optimization steps.
        ef_coef_base (float, optional): Base coefficient for formation energy loss. Default is 1.0.
        T_min (float, optional): Minimum temperature. Default is 0.0001.
        T_max (float, optional): Maximum temperature. Default is 0.01.

    Returns:
        Tuple[List[float], List[float], float]:
            - dist_temp_sche: Temperature schedule (decreasing).

    """
    dist_temp_sche = [
        T_min + (T_max - T_min) / num_steps * i for i in range(num_steps + 1)
    ][::-1]

    return dist_temp_sche
