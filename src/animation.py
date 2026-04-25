"""Animation helpers for vehicle movement and turning."""

import math
import pygame
from src.config import VEHICLE_LENGTH, VEHICLE_WIDTH, SUSPENSION_AMP, RED, WHITE, BLACK


def bezier_point(p0, p1, p2, t):
    """Quadratic bezier point at parameter t."""
    (x0, y0), (x1, y1), (x2, y2) = p0, p1, p2
    u = 1 - t
    x = u * u * x0 + 2 * u * t * x1 + t * t * x2
    y = u * u * y0 + 2 * u * t * y1 + t * t * y2
    return (x, y)


def bezier_derivative(p0, p1, p2, t):
    """Derivative (tangent) of quadratic bezier at t."""
    (x0, y0), (x1, y1), (x2, y2) = p0, p1, p2
    u = 1 - t
    dx = 2 * u * (x1 - x0) + 2 * t * (x2 - x1)
    dy = 2 * u * (y1 - y0) + 2 * t * (y2 - y1)
    return (dx, dy)


def heading_from_deltas(dx, dy):
    """Compute heading angle from direction vector."""
    if dx == 0 and dy == 0:
        return 0.0
    return math.atan2(dy, dx)


def interpolate_angle(a, b, t):
    """Smoothly interpolate between two angles handling wraparound."""
    diff = b - a
    while diff > math.pi:
        diff -= 2 * math.pi
    while diff < -math.pi:
        diff += 2 * math.pi
    return a + diff * t


def create_vehicle_surface(color, heading):
    """Create rotated vehicle surface with car-like shape."""
    surf = pygame.Surface((VEHICLE_LENGTH + 4, VEHICLE_WIDTH + 4), pygame.SRCALPHA)
    surf.fill((0, 0, 0, 0))

    points = [
        (-VEHICLE_LENGTH // 2, -VEHICLE_WIDTH // 2 + 2),
        (VEHICLE_LENGTH // 2 - 4, -VEHICLE_WIDTH // 2 + 2),
        (VEHICLE_LENGTH // 2, 0),
        (VEHICLE_LENGTH // 2 - 4, VEHICLE_WIDTH // 2 - 2),
        (-VEHICLE_LENGTH // 2, VEHICLE_WIDTH // 2 - 2),
    ]

    pygame.draw.polygon(surf, color, [(p[0] + VEHICLE_LENGTH // 2 + 2, p[1] + VEHICLE_WIDTH // 2 + 2) for p in points])
    pygame.draw.polygon(surf, BLACK, [(p[0] + VEHICLE_LENGTH // 2 + 2, p[1] + VEHICLE_WIDTH // 2 + 2) for p in points], 1)

    return surf, (VEHICLE_LENGTH // 2 + 2, VEHICLE_WIDTH // 2 + 2)


def draw_brake_lights(surf, pos, heading):
    """Draw brake lights at rear of vehicle."""
    cos_h = math.cos(heading)
    sin_h = math.sin(heading)

    offset_x = -VEHICLE_LENGTH // 2 * cos_h
    offset_y = -VEHICLE_LENGTH // 2 * sin_h

    rear_x = pos[0] + offset_x
    rear_y = pos[1] + offset_y

    perp_x = -sin_h * 3
    perp_y = cos_h * 3

    pygame.draw.circle(surf, RED, (int(rear_x - perp_x), int(rear_y - perp_y)), 2)
    pygame.draw.circle(surf, RED, (int(rear_x + perp_x), int(rear_y + perp_y)), 2)


def draw_accel_lines(surf, pos, heading, velocity):
    """Draw acceleration lines behind vehicle."""
    if velocity < 5:
        return
    cos_h = math.cos(heading)
    sin_h = math.sin(heading)

    base_x = pos[0] - cos_h * VEHICLE_LENGTH // 2
    base_y = pos[1] - sin_h * VEHICLE_LENGTH // 2

    for i in range(3):
        offset = (i - 1) * 4
        perp_x = -sin_h * offset
        perp_y = cos_h * offset

        start_x = base_x + perp_x - cos_h * 5
        start_y = base_y + perp_y - sin_h * 5
        end_x = start_x - cos_h * 10
        end_y = start_y - sin_h * 10

        pygame.draw.line(surf, WHITE, (int(start_x), int(start_y)), (int(end_x), int(end_y)), 1)


def suspension_offset(phase):
    """Compute Y-axis suspension bounce offset."""
    return math.sin(phase) * SUSPENSION_AMP