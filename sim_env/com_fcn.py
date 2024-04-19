import pygame
from parameters import PIXEL_TO_METER_SCALE, COLORS


def meters_to_pixels(meters):
    """
    Convert meters to pixels based on the defined scale.

    Parameters:
        meters: The value in meters to convert.

    Returns:
        float: The equivalent value in pixels.
    """
    return meters / PIXEL_TO_METER_SCALE


def draw_object(screen, color, vertex):
    pixel_vertex = meters_to_pixels(vertex)
    pygame.draw.polygon(screen, COLORS[color], pixel_vertex)
