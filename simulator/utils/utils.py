# ----------------------------------------------------
# Second edition of HAADS(used for RL simulation)
# Author: Xiangyu Liu
# Date: 2016.11.26
# Filename: utils.py
# ----------------------------------------------------

import math
import numpy as np
import itertools
import random

import os
import shutil
import glob

def act_cost(a):
    return (a % 2 == 0 and a != 0) and 0.01 * math.sqrt(2) or 0.01


def gen_margin(map_size, size, diff, pos):
    def legal_map(m_size, pos):
        if 0 <= pos[0] < m_size and 0 <= pos[1] < m_size:
            return True
        else:
            return False

    xmin, xmax = pos[0] - diff, pos[0] + diff
    ymin, ymax = pos[1] - diff, pos[1] + diff
    result = []
    result += list(itertools.product(range(xmin, xmin+1), range(ymin, ymax+1)))
    result += list(itertools.product(range(xmax, xmax+1), range(ymin, ymax+1)))
    result += list(itertools.product(range(xmin, xmax+1), range(ymin, ymin+1)))
    result += list(itertools.product(range(xmin, xmax+1), range(ymax, ymax+1)))
    result = list(set(result))
    result = [item for item in result if legal_map(map_size, item)]
    return result


def delete_folder(src):
    '''delete files and folders'''
    f_list = glob.glob(src + '*')
    for i in f_list:
        if os.path.isdir(i) and i.lstrip(src).isdigit():
            shutil.rmtree(i)


def fmt_item(x, l):
    if isinstance(x, np.ndarray):
        assert x.ndim == 0
        x = x.item()
    if isinstance(x, float):
        rep = "%g" % x
    else:
        rep = str(x)
    return " " * (l - len(rep)) + rep


def fmt_row(width, row):
    out = " | ".join(fmt_item(x, width) for x in row)
    return out


def rect_collision(item1, item2):
    # Just for drawing map(ob & wall)
    # spare = 1
    spare = 0
    x1, y1 = item1.pos_x, item1.pos_y
    try:
        w1, l1 = item1.size, item1.size
    except:
        w1, l1 = item1.width, item1.length

    x2, y2 = item2.pos_x, item2.pos_y
    try:
        w2, l2 = item2.size, item2.size
    except:
        w2, l2 = item2.width, item2.length

    if (x2 + w2 < x1 - spare) or (x1 + w1 < x2 - spare) or \
                    (y1 + l1 < y2 - spare) or (y2 + l2 < y1 - spare):
        return False
    return True


def circle_collision(item1, item2):
    # Just for drawing agent
    cx1, cy1, cr1 = item1.pos_x, item1.pos_y, item1.radius
    cx2, cy2, cr2 = item2.pos_x, item2.pos_y, item2.radius

    distance = math.sqrt((cx2 - cx1) ** 2 + (cy2 - cy1) ** 2)
    if distance < cr1 + cr2:
        return True
    else:
        return False


def rect_circle_collision(rect_item, circle_item):
    # For agent motion
    # spare = circle_item.velocity * 0.01

    spare = 0
    # print spare

    try:
        rw, rl = rect_item.size, rect_item.size
        rx, ry = rect_item.pos_x + rect_item.size / 2, rect_item.pos_y + rect_item.size / 2
    except:
        rw, rl = rect_item.width, rect_item.length
        rx, ry = rect_item.pos_x + rect_item.width / 2, rect_item.pos_y + rect_item.length / 2

    cx, cy, cr = circle_item.pos_x, circle_item.pos_y, circle_item.radius

    circle_distance_x = abs(cx - rx)
    circle_distance_y = abs(cy - ry)
    if circle_distance_x > rw / 2.0 + cr or circle_distance_y > rl / 2.0 + cr:
        return False
    if circle_distance_x <= rw / 2.0 or circle_distance_y <= rl / 2.0:
        return True
    corner_x = circle_distance_x - rw / 2.0
    corner_y = circle_distance_y - rl / 2.0
    corner_distance_sq = corner_x ** 2.0 + corner_y ** 2.0
    return corner_distance_sq <= cr ** 2.0 + spare