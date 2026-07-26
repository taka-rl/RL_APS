import pygame
from sim_env.parameters import PIXEL_TO_METER_SCALE, PI, RenderConfig, WheelSize
from sim_env.car import Car


def meters_to_pixels(meters):
    """
    Convert meters to pixels based on the defined scale.

    Parameters:
        meters: The value in meters to convert.

    Returns:
        float: The equivalent value in pixels.
    """
    return meters / PIXEL_TO_METER_SCALE


class Renderer:
    """
    Handles all rendering operations in the parking environment.

    Attributes:
        window (pygame.Surface): The main Pygame window where the environment is rendered.
        surf_car (pygame.Surface): A surface object representing the car (agent) in the environment.
        surf_text (pygame.Surface): A transparent surface used for rendering text overlays.
        surf_parkinglot (pygame.Surface): A surface object representing the parking lot, static obstacles, and grid.
        clock (pygame.time.Clock): A Pygame clock object for controlling the frame rate.
        font (pygame.font.Font): The font used for rendering text information on the screen.
        fps (int): The frames per second (FPS) setting for rendering.
        grid_size (int): The size of the grid lines in the parking lot (in pixels).
        colors (dict): A dictionary of colors used for rendering different elements in the environment.
        wheel_size (WheelSize): The dimensions of the car's wheels, used for rendering them correctly.

    """
    def __init__(self, render_config: RenderConfig, wheel_size: WheelSize) -> None:
        self.window = None
        self.clock = None
        self.surf_parkinglot = None
        self.surf_car = None
        self.surf_text = None
        self.font = None

        # Extract rendering configurations
        self.fps = render_config.fps
        self.window_width = render_config.window_width
        self.window_height = render_config.window_height
        self.grid_size = render_config.grid_size
        self.colors = render_config.colors

        self.wheel_size = wheel_size

    def initialize_window(self) -> None:
        """Initialize the Pygame window and necessary components."""
        # Initialize the parking environment window
        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_width, self.window_height))
            pygame.display.set_caption('Parking Environment')

        # Need to understand how and why I need this.
        if self.clock is None:
            self.clock = pygame.time.Clock()

        # Initialize the text display
        if self.surf_text is None:
            pygame.font.init()
            self.font = pygame.font.SysFont('Times New Roman', 15)
            self.surf_text = pygame.Surface((self.window_width, self.window_height), flags=pygame.SRCALPHA)

    def draw_static_elements(self, parking_lot_vertices, static_parking_lot_vertices, static_cars_vertices) -> None:
        """Create the parking lot surface once (cached)."""
        # Initialize the parking lot surface
        if self.surf_parkinglot is None:
            self.surf_parkinglot = pygame.Surface((self.window_width, self.window_height), flags=pygame.SRCALPHA)
            self.surf_parkinglot.fill(self.colors['WHITE'])

            # Draw grid lines
            for x in range(0, self.window_width, self.grid_size):
                pygame.draw.line(self.surf_parkinglot, self.colors['GRID_COLOR'], (x, 0),
                                 (x, self.window_height))
            for y in range(0, self.window_height, self.grid_size):
                pygame.draw.line(self.surf_parkinglot, self.colors['GRID_COLOR'], (0, y),
                                 (self.window_width, y))

            # Draw the parking lot
            self.draw_object(self.surf_parkinglot, self.colors["RED"], parking_lot_vertices)

            # Draw static obstacles
            for parking_lot_vertex in static_parking_lot_vertices:
                self.draw_object(self.surf_parkinglot, self.colors["YELLOW"], parking_lot_vertex)
            for car_vertex in static_cars_vertices:
                self.draw_object(self.surf_parkinglot, self.colors["GREY"], car_vertex)

    def draw_text_display(self, car_loc, v, psi) -> None:
        """Draw text display"""
        # Clear previous text
        self.surf_text.fill((0, 0, 0, 0))

        # Display car status text
        text_str = (f'Car location: X: {car_loc[0]:.4f}, Y: {car_loc[1]:.4f}\n'
                    f'Velocity: {v:.4f}m/s\n'
                    f'Heading angle: {psi:.4f}rad\n'
                    f'Degree: {psi * (180 / PI):.4f}degree')

        # Define the rectangle area for the text display
        text_rect = pygame.Rect(400, 500, 100, 100)
        self.draw_multiline_text(self.surf_text, text_str, self.colors['BLACK'], text_rect, self.font)

    def draw_car_path(self, car_loc_old, car_loc) -> None:
        """Draw the car path"""
        # Draw path (line between previous and current position)
        car_loc_old_pixels = meters_to_pixels(car_loc_old)
        car_loc_pixels = meters_to_pixels(car_loc)
        pygame.draw.line(self.surf_parkinglot, self.colors['BLACK'], car_loc_old_pixels, car_loc_pixels)

    def draw_car(self, car: Car, car_loc) -> None:
        """Draw the car(agent)"""
        if self.surf_car is None:
            self.surf_car = pygame.Surface((self.window_width, self.window_height),
                                           flags=pygame.SRCALPHA)

        # Clear the car surface
        self.surf_car.fill((0, 0, 0, 0))

        # Draw the car(agent)
        self.draw_object(self.surf_car, self.colors["GREEN"], car.calc_car_vertices())

        # wheels
        # calculate the rotation of the wheels
        wheel_points = car.rotate_car(self.wheel_size.wheel_pos, angle=car.psi)
        # draw each wheel
        for i, wheel_point in enumerate(wheel_points):
            if i < 2:
                wheel_vertices = car.rotate_car(self.wheel_size.wheel_struct, angle=car.psi + car.delta)
            else:
                wheel_vertices = car.rotate_car(self.wheel_size.wheel_struct, angle=car.psi)
            wheel_vertices += wheel_point + car_loc
            self.draw_object(self.surf_car, self.colors["RED"], wheel_vertices)

    def draw_dynamic_elements(self, car: Car, car_loc, car_loc_old) -> None:
        """Draw car, movement path, and text updates."""
        self.draw_car(car, car_loc)
        self.draw_car_path(car_loc_old, car_loc)
        self.draw_text_display(car_loc, car.v, car.psi)

    def render(self, car: Car, car_loc_old) -> None:
        """Render the environment with updated car position."""
        # Draw the dynamic objects
        self.draw_dynamic_elements(car, car.car_loc, car_loc_old)

        # Compose final frame
        surf = self.surf_parkinglot.copy()
        surf.blit(self.surf_car, (0, 0))
        surf = pygame.transform.flip(surf, False, True)
        surf.blit(self.surf_text, (0, 0))

        # Update display
        pygame.event.pump()
        self.clock.tick(self.fps)
        self.window.fill(self.colors['BLACK'])
        self.window.blit(surf, (0, 0))
        pygame.display.flip()

    def reset_render(self):
        """Reset renderer class attributes"""
        self.surf_parkinglot = None
        self.surf_car = None

    def close(self) -> None:
        """ Close the Pygame window and clean up resources. """
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    @staticmethod
    def draw_object(screen: pygame.Surface, color: tuple, vertex) -> None:
        """Draw an object using a list of vertices."""
        pixel_vertex = meters_to_pixels(vertex)
        pygame.draw.polygon(screen, color, pixel_vertex)

    @staticmethod
    def draw_multiline_text(screen: pygame.Surface, text: str, color: tuple,
                            rect: pygame.Rect, font: pygame.font) -> None:
        """Render multi-line text onto a surface."""
        lines = text.splitlines()
        y = rect.top
        for line in lines:
            line_surface = font.render(line, True, color)
            screen.blit(line_surface, (rect.left, y))
            y += line_surface.get_height()
