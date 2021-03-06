import numpy as np
from base import Base, ACTION_LIST

OBSTACLE_COLOR = (128, 128, 128)


class Obstacle(Base):

    def __init__(
            self,
            identity,
            pos_x, pos_y,
            size, tmp_vector, color=OBSTACLE_COLOR
    ):
        super(Obstacle, self).__init__(identity, pos_x, pos_y, size, color)

        # pay attention: tuple doesn't support assignment

        # tmp_vector = (0, 0)
        self.vector[0] = tmp_vector[0]
        self.vector[1] = tmp_vector[1]

    def step_back(self):
        super(Obstacle, self).step_back()

    def collide_walls(self, screen_rect):
        return super(Obstacle, self).collide_walls(screen_rect)

    def switch_components(self, other, i):
        super(Obstacle, self).switch_components(other, i)

    def collide(self, others):
        return super(Obstacle, self).collide(others)

    def update(self, screen_rect, others, info):
        super(Obstacle, self).update(screen_rect, others, info)

    @classmethod
    def random(cls, map_size, size, seed=None):
        if seed:
            np.random.seed(seed)
        pos_x = np.random.randint(0 + size / 2, map_size - size / 2)
        pos_y = np.random.randint(0 + size / 2, map_size - size / 2)
        tmp_vector = tuple(np.asarray(ACTION_LIST)[np.random.choice(len(ACTION_LIST), 1, replace=False)][0])
        return pos_x, pos_y, tmp_vector
