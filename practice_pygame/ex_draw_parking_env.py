import pygame
import sys
import numpy as np

# Initialize Pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 800, 600
WHITE = (255, 255, 255)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
GRID_COLOR = (200, 200, 200)
GRID_SIZE = 20
WALL_COLOR = (0, 0, 255)
GREEN = (0, 255, 0)

# Car and wheel
CAR_L = 80
CAR_W = 40
WHEEL_LENGTH = 15
WHEEL_WIDTH = 7
WHEEL_POS = np.array([[25, 15], [25, -15], [-25, -15], [-25, 15]])
CAR_LOC = [300, 100]
PARKING_LOT_LOC = [300, 10, CAR_L+10, CAR_W+10]

# Car class
class Car(pygame.sprite.Sprite):
    def __init__(self, color, width, height):
        super().__init__()

        # Create a car surface and fill it with the specified color
        self.image = pygame.Surface([width, height])
        self.image.fill(color)

        # Set the rectangle (position and dimensions) of the car
        self.rect = self.image.get_rect()


    def draw_vehicle(self, screen, car_loc, psi):
        # the car(agent)
        # get the car(agent) vertexes from the center point
        car_vertices = self.compute_vertices(car_loc, CAR_L, CAR_W)
        # get the rotated car location
        car_vertices = self.rotate_car(car_vertices, angle=psi)
        # draw the car(agent)
        pygame.draw.polygon(screen, GREEN, car_vertices)

        # wheels
        # get the center of each wheel point
        wheel_points = self.compute_wheel_points(car_loc)
        # draw each wheel
        for i, wheel_point in enumerate(wheel_points):
            wheel_vertices = self.compute_vertices(wheel_point, WHEEL_LENGTH, WHEEL_WIDTH)
            if i < 2:
                delta = 0.003
                wheel_vertices = self.rotate_car(wheel_vertices, angle=psi + delta)
            else:
                wheel_vertices = self.rotate_car(wheel_vertices, angle=psi)
            pygame.draw.polygon(screen, RED, wheel_vertices)


    def compute_vertices(self, car_loc, length, width):
        vertices = np.array([
            [car_loc[0] + length / 2, car_loc[1] + width / 2],
            [car_loc[0] + length / 2, car_loc[1] - width / 2],
            [car_loc[0] - length / 2, car_loc[1] - width / 2],
            [car_loc[0] - length / 2, car_loc[1] + width / 2]
        ])
        return vertices

    def compute_wheel_points(self, car_loc):
        wheel_points = np.array([
            [car_loc[0] + WHEEL_POS[0, 0], car_loc[1] + WHEEL_POS[0, 1]],
            [car_loc[0] + WHEEL_POS[1, 0], car_loc[1] + WHEEL_POS[1, 1]],
            [car_loc[0] + WHEEL_POS[2, 0], car_loc[1] + WHEEL_POS[2, 1]],
            [car_loc[0] + WHEEL_POS[3, 0], car_loc[1] + WHEEL_POS[3, 1]]
        ])
        return wheel_points

    def rotate_car(self, pos, angle=0):
        R = np.array([
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)],
        ])
        rotated_pos = (R @ pos.T).T

        return rotated_pos





# Initialize the game window
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Parking environment")

# Create a group to hold all the sprites (cars)
all_sprites = pygame.sprite.Group()

# Create cars
target_car = Car(GREEN, CAR_L, CAR_W)
car1 = Car(RED, CAR_L, CAR_W)
car2 = Car(RED, CAR_L, CAR_W)

# Set initial positions for the cars
target_car.rect.x = 300
target_car.rect.y = 100

car1.rect.x = 100
car1.rect.y = 200

car2.rect.x = 200
car2.rect.y = 200

# Add cars to the sprite group
# all_sprites.add(target_car)
all_sprites.add(car1)
all_sprites.add(car2)

# Main game loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Clear the screen
    screen.fill(WHITE)

    # Draw the grid lines
    for x in range(0, WIDTH, GRID_SIZE):
        pygame.draw.line(screen, GRID_COLOR, (x, 0), (x, HEIGHT), 1)
    for y in range(0, HEIGHT, GRID_SIZE):
        pygame.draw.line(screen, GRID_COLOR, (0, y), (WIDTH, y), 1)

    # Draw the parking lot
    pygame.draw.rect(screen, YELLOW, PARKING_LOT_LOC)

    # Draw the wall
    pygame.draw.rect(screen, WALL_COLOR, [200, 0, 20, 200])

    # Update and draw all sprites
    all_sprites.update()
    all_sprites.draw(screen)

    # draw the agent
    psi = 0.05
    target_car.draw_vehicle(screen, CAR_LOC, psi)

    # Draw wheels for the target car
    # target_car.draw_wheels(screen)

    # Update the display
    pygame.display.flip()

    # Control the frame rate
    pygame.time.Clock().tick(60)

# Quit Pygame
pygame.quit()
sys.exit()
