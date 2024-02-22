"""
This is an example code to learn how to draw a perpendicular parking environment using Pygame.
"""

import pygame
import sys

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

# Car class
class Car(pygame.sprite.Sprite):
    def __init__(self, color, width, height):
        super().__init__()

        # Create a car surface and fill it with the specified color
        self.image = pygame.Surface([width, height])
        self.image.fill(color)

        # Set the rectangle (position and dimensions) of the car
        self.rect = self.image.get_rect()

# Initialize the game window
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Perpendicular Parking")

# Create a group to hold all the sprites (cars)
all_sprites = pygame.sprite.Group()

# Create cars
target_car = Car(GREEN, 50, 30)
car1 = Car(RED, 50, 30)
car2 = Car(RED, 50, 30)

# Set initial positions for the cars
target_car.rect.x = 300
target_car.rect.y = 100

car1.rect.x = 100
car1.rect.y = 200

car2.rect.x = 200
car2.rect.y = 200

# Add cars to the sprite group
all_sprites.add(target_car)
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
    pygame.draw.rect(screen, YELLOW, [300, 10, 50, 30]) # [X, Y, Width, Height]

    # Draw the wall
    pygame.draw.rect(screen, WALL_COLOR, [200, 0, 20, 200]) # [X, Y, Width, Height]

    # Update and draw all sprites
    all_sprites.update()
    all_sprites.draw(screen)

    # Update the display
    pygame.display.flip()

    # Control the frame rate
    pygame.time.Clock().tick(60)

# Quit Pygame
pygame.quit()
sys.exit()
