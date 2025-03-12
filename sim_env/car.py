import numpy as np
from sim_env.com_fcn import draw_object
from sim_env.parameters import Config


class Car:
    def __init__(self, car_loc, psi, config: Config):
        self.car_loc = car_loc
        self.psi = psi
        self.v = 0.0
        self.delta = 0.0
        self.config = config
        self.wheel_size = self.config.wheel_size
        self.car_size = self.config.car_size
        self.car_vertices = self.calc_car_vertices()

    def kinematic_act(self, action):
        """
        Calculate the car(agent) movement

        Parameters:
            action(list): [a, δ]: a is acceleration, δ(delta) is steering angle.
            self.v : velocity
            self.psi(ψ): the heading angle of the car

        Kinematic bicycle model:
        x_dot = v * np.cos(psi)
        y_dot = v * np.sin(psi)
        v_dot = a
        psi_dot = v * np.tan(delta) / CAR_L
        """
        x_dot = self.v * np.cos(self.psi)
        y_dot = self.v * np.sin(self.psi)
        v_dot = action[0]
        psi_dot = self.v * np.tan(action[1]) / self.car_size.length
        car_loc = np.array([x_dot, y_dot])
        self.update_state(car_loc, v_dot, psi_dot, self.config.dt)
        self.delta = action[1]
        self.car_vertices = self.calc_car_vertices()

    def update_state(self, car_loc, v_dot, psi_dot, dt):
        self.car_loc += dt * car_loc
        self.v = np.clip(self.v + v_dot, -self.config.velocity_limit, self.config.velocity_limit)
        self.psi += dt * psi_dot

    @staticmethod
    def rotate_car(car_loc, angle=0.0) -> np.array:
        """
        Rotate vertices by a given angle.

        Parameters:
            car_loc (np.array): The car's vertices to rotate.
            angle (float): Rotation angle in radians.

        Returns:
            np.array: Rotated vertices.
        """
        r = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)],
        ])
        return (r @ car_loc.T).T

    def calc_car_vertices(self) -> np.array:
        """
        Calculate the car vertices

        Return:
            np.array: car vertices
        """
        return self.rotate_car(self.car_size.car_struct, angle=self.psi) + self.car_loc

    def draw_car(self, screen):
        """
        Draw the car(agent) and its wheel

        Parameters:
            screen: pygame.Surface
        """
        # draw the car(agent)
        draw_object(screen, self.config.colors["GREEN"], self.car_vertices)

        # wheels
        # calculate the rotation of the wheels
        wheel_points = self.rotate_car(self.wheel_size.wheel_pos, angle=self.psi)
        # draw each wheel
        for i, wheel_point in enumerate(wheel_points):
            if i < 2:
                wheel_vertices = self.rotate_car(self.wheel_size.wheel_struct, angle=self.psi + self.delta)
            else:
                wheel_vertices = self.rotate_car(self.wheel_size.wheel_struct, angle=self.psi)
            wheel_vertices += wheel_point + self.car_loc
            draw_object(screen, self.config.colors["RED"], wheel_vertices)
