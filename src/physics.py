"""Physics models: IDM car-following and MOBIL lane-changing."""

import math
from src.config import IDM_A_MAX, IDM_B, IDM_V0, IDM_S0, IDM_T, MOBIL_P, MOBIL_THRESHOLD, MOBIL_B_SAFE


def idm_acceleration(v: float, dv: float, s: float) -> float:
    """Intelligent Driver Model acceleration."""
    if s <= 0:
        return -IDM_B * 10
    s_star = IDM_S0 + max(0, v * IDM_T + v * dv / (2 * math.sqrt(IDM_A_MAX * IDM_B)))
    a = IDM_A_MAX * (1 - (v / IDM_V0) ** 4 - (s_star / s) ** 2)
    return max(-IDM_B * 5, a)


def mobil_incentive(a_cur: float, a_new: float, a_surrounding: float) -> float:
    """MOBIL lane change incentive."""
    return (a_new - a_cur) + MOBIL_P * (a_surrounding - IDM_A_MAX * 0.5)


def mobil_safe(a_new_lane_lead: float) -> bool:
    """Check if lane change is safe."""
    return a_new_lane_lead > -MOBIL_B_SAFE