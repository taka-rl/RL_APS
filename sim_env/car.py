import numpy as np
from sim_env.parameters import Config


class Car:
    def __init__(self, car_loc, loc_old, psi, config: Config):
        self.car_loc = car_loc
        self.loc_old = loc_old
        self.psi = psi
        self.v = 0.0
        self.delta = 0.0
        self.config = config
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
        self.v = np.clip(self.v + dt * v_dot, -self.config.velocity_limit, self.config.velocity_limit)
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
