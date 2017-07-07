# ----------------------------------------------------
# Second edition of HAADS(used for RL simulation)
# Author: Xiangyu Liu
# Date: 2016.11.26
# Filename: collision_detect.py
# ----------------------------------------------------

from utils import *
import numpy as np
import pygame as pg


def collide_other(one, two):
    """
    Callback function for use with pg.sprite.collidesprite methods.
    It simply allows a sprite to test collision against its own group,
    without returning false positives with itself.
    """
    return one is not two and pg.sprite.collide_rect(one, two)


def pattern_match(state):
    if type(state).__name__ == "obstacle_state":
        return "rect"
    if type(state).__name__ == "checkpoint_state":
        return "rect"
    if type(state).__name__ == "wall_state":
        return "rect"
    if type(state).__name__ == "agent_state":
        return "rect"


def collision_detect(cover_set, state):
    assert isinstance(cover_set, list)
    for item in cover_set:
        if pattern_match(item) == "rect" and pattern_match(state) == "rect":
            if rect_collision(item, state):
                return True
        elif pattern_match(item) == "circle" and pattern_match(state) == "circle":
            if circle_collision(item, state):
                return True
        elif pattern_match(item) == "circle" and pattern_match(state) == "rect":
            if rect_circle_collision(state, item):
                return True
        elif pattern_match(item) == "rect" and pattern_match(state) == "circle":
            if rect_circle_collision(item, state):
                return True
        else:
            raise NotImplementedError
    return False


def bounce(self_state, ob_state, decay):
    # spare = self_state.velocity * 0.02
    theta_return = 0
    if type(ob_state).__name__ == "agent_state":
        v1, v2, theta1, theta2 = self_state.velocity, ob_state.velocity, self_state.theta, ob_state.theta
        v1 = np.array([v1 * math.cos(math.radians(theta1)),
                       v1 * math.sin(math.radians(theta1))])
        v2 = np.array([v2 * math.cos(math.radians(theta2)),
                       v2 * math.sin(math.radians(theta2))])
        pos1 = np.array([self_state.pos_x + self_state.size / 2,
                         self_state.pos_y + self_state.size / 2])
        pos2 = np.array([ob_state.pos_x + ob_state.size / 2,
                         ob_state.pos_y + ob_state.size / 2])

        v1_new = v1 - np.inner(v1 - v2, pos1 - pos2) / \
            np.sum(np.square(pos1 - pos2)) * (pos1 - pos2)
        theta_return = math.degrees(math.atan2(v1_new[1], v1_new[0]))
    else:
        # get self information
        pos_x1, pos_y1, theta1 = self_state.pos_x, self_state.pos_y, self_state.theta
        try:
            size1 = self_state.size
        except:
            size1 = self_state.width
        # get collision information
        pos_x2, pos_y2 = ob_state.pos_x, ob_state.pos_y
        try:
            w2, l2 = ob_state.width, ob_state.length
        except:
            w2, l2 = ob_state.size, ob_state.size

        assert 0 <= theta1 < 360

        if theta1 <= 90:
            if pos_x1 + size1 > pos_x2 > pos_x1:
                theta_return = 180 - theta1
            elif pos_y1 + size1 > pos_y2 > pos_y1:
                theta_return = 360 - theta1
            else:
                print "1.exception"
                # print self_state, ob_state
                theta_return = (180 + theta1) % 360
        elif theta1 <= 180:
            if pos_x1 < pos_x2 + w2 < pos_x1 + size1:
                theta_return = 180 - theta1
            elif pos_y1 + size1 > pos_y2 > pos_y1:
                theta_return = 360 - theta1
            else:
                print "2.exception"
                # print self_state, ob_state
                theta_return = (180 + theta1) % 360
        elif theta1 <= 270:
            if pos_x1 < pos_x2 + w2 < pos_x1 + size1:
                theta_return = 540 - theta1
            elif pos_y1 < pos_y2 + l2 < pos_y1 + size1:
                theta_return = 360 - theta1
            else:
                print "3.exception"
                # print self_state, ob_state
                theta_return = (180 + theta1) % 360
        else:
            if pos_x1 + size1 > pos_x2 > pos_x1:
                theta_return = 540 - theta1
            elif pos_y1 < pos_y2 + l2 < pos_y1 + size1:
                theta_return = 360 - theta1
            else:
                print "4.exception"
                # print self_state, ob_state
                theta_return = (180 + theta1) % 360
        try:
            velocity = self_state.velocity * decay
        except:
            velocity = None

    return velocity, theta_return


def wall_collision(state, wall, decay):
    pos_x, pos_y = state.pos_x, state.pos_y
    try:
        size = state.size
    except:
        size = state.width

    out_left = pos_x < wall.internal_range.xmin
    out_right = pos_x + size > wall.internal_range.xmax
    out_top = pos_y < wall.internal_range.ymin
    out_bottom = pos_y + size > wall.internal_range.ymax

    tmp_x = math.cos(math.radians(state.theta))
    tmp_y = math.sin(math.radians(state.theta))

    if out_left or out_right:
        tmp_x *= -1
    if out_top or out_bottom:
        tmp_y *= -1
    new_theta = math.degrees(math.atan2(tmp_y, tmp_x))
    try:
        new_v = decay * state.velocity
    except:
        new_v = None

    return (out_left, out_right, out_top, out_bottom), new_v, new_theta
