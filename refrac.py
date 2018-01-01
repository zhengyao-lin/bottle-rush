#! /usr/bin/python3

import numpy as np
from PIL import Image
import math
import time
import adb as pyadb3
import cv2
import os
import io

INF = float("inf")

class Util:
    @staticmethod
    def resize(img, ratio):
        h, w = img.shape[:2]
        return cv2.resize(img, (int(w * ratio), int(h * ratio)))

    @staticmethod
    def pin(img, pos, radius = 3, color = (0, 0, 0)):
        cv2.circle(img, pos, radius, color, -1)

# low-level interaction with the device
class Device:
    PRESS_POINT = (600, 600)
    RESIZE = 0.3 # resize the input image for better performance

    def __init__(self):
        self.adb = pyadb3.ADB()

        h, w = self.screencap().shape[:2]
        Device.PRESS_POINT = (w / 2 / Device.RESIZE, h / 2 / Device.RESIZE)

    def screencap(self):
        self.adb.shell_command("screencap -p")
        raw = self.adb.get_output()
        img = cv2.imdecode(np.fromstring(raw, np.uint8), cv2.IMREAD_COLOR)

        img = Util.resize(img, Device.RESIZE)

        return img

    def press(self, duration): # duration in ms
        x, y = Device.PRESS_POINT
        cmd = "input swipe %d %d %d %d %d" % (x, y, x, y, duration)
        print(cmd)
        self.adb.shell_command(cmd)

# image -> start & end point
class Marker:
    CENTER_DELTA = (1.45631, 1.13) # in TJSU
    BOTTLE_HEIGHT = 10 # constant in the game = head_pos + head_radius
    UNIT = -1 # pixel / TJSU

    JUMP_RIGHT_ANGLE_TAN = 0.579
    JUMP_LEFT_ANGLE_TAN = 0.555

    JUMP_RIGHT_ANGLE_SIN = JUMP_RIGHT_ANGLE_TAN / ((1 + JUMP_RIGHT_ANGLE_TAN ** 2) ** 0.5)
    JUMP_LEFT_ANGLE_SIN = JUMP_LEFT_ANGLE_TAN / ((1 + JUMP_LEFT_ANGLE_TAN ** 2) ** 0.5)

    JUMP_RIGHT_ANGLE_COS = JUMP_RIGHT_ANGLE_SIN / JUMP_RIGHT_ANGLE_TAN
    JUMP_LEFT_ANGLE_COS = JUMP_LEFT_ANGLE_SIN / JUMP_LEFT_ANGLE_TAN

    JUMP_RIGHT_ANGLE = math.asin(JUMP_RIGHT_ANGLE_SIN)
    JUMP_LEFT_ANGLE = math.asin(JUMP_LEFT_ANGLE_SIN)

    def __init__(self, path):
        self.bottle = cv2.imread(path, 0)
        Marker.UNIT = self.bottle.shape[0] / Marker.BOTTLE_HEIGHT
        self.prev_dir = 1

    def find_bottle(self, screen):
        h, w = self.bottle.shape

        res = cv2.matchTemplate(cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY),
                                self.bottle, cv2.TM_CCOEFF_NORMED)

        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        pos = (int(max_loc[0] + w / 2), int(max_loc[1] + h * 0.9))

        cv2.rectangle(screen, max_loc, (max_loc[0] + w, max_loc[1] + h), (0, 0, 0), 1)
        # cv2.rectangle(screen, min_loc, (min_loc[0] + w, min_loc[1] + h), (0, 0, 255), 2)
        Util.pin(screen, pos)
        # cv2.imshow("screen", screen)

        # pos -> center of the bottle
        return pos, max_val

    # calib :: screen -> update self.bottle Marker.UNIT
    def calib(self, screen, scale = 0):
        max_val = -INF
        max_scale = scale

        if scale == 0:
            for scale in np.linspace(0.2, 4, 30).tolist():
                print("trying scale %f" % scale)

                nscreen = Util.resize(screen, scale)   
                _, val = self.find_bottle(nscreen)

                if val > max_val:
                    max_val = val
                    max_scale = scale
        
        print("optimal scale %f" % max_scale)
        
        self.bottle = Util.resize(self.bottle, 1 / max_scale)
        Marker.UNIT = self.bottle.shape[0] / Marker.BOTTLE_HEIGHT

        return max_scale

    def next(self, screen, bottle_pos):
        h, w = screen.shape[:2]
        bx, by = bottle_pos

        cx, cy = int((w + Marker.CENTER_DELTA[0] * Marker.UNIT) / 2), \
                 int((h + Marker.CENTER_DELTA[1] * Marker.UNIT) / 2)

        # cv2.circle(screen, (cx, cy), 4, (100, 100, 100), -1)
        Util.pin(screen, (cx, cy))

        orig = dist = ((cx - bx) ** 2 + (cy - by) ** 2) ** 0.5

        # correction on turn

        if (cx - bx) * self.prev_dir < 0: # different direction as the previous jump
            if self.prev_dir == 1: # now turn to the left
                a1 = math.atan(abs(cy - by) / abs(cx - bx)) + Marker.JUMP_RIGHT_ANGLE
            else: # now turn to the right
                a1 = math.atan(abs(cy - by) / abs(cx - bx)) + Marker.JUMP_LEFT_ANGLE
            
            d1 = dist * math.sin(a1)
            tot = Marker.JUMP_LEFT_ANGLE + Marker.JUMP_RIGHT_ANGLE

            if tot < math.pi / 2:
                dist = d1 / math.sin(tot)
            else:
                dist = d1 / math.sin(math.pi - tot)

        # print(orig, "->", dist)

        if cx > bx: # jump right
            # dist = math.sin(ang) * dist
            self.prev_dir = 1
            px = cx + dist * Marker.JUMP_RIGHT_ANGLE_COS
            py = cy - dist * Marker.JUMP_RIGHT_ANGLE_SIN
        else: # jump left
            self.prev_dir = -1
            # dist = math.sin(ang) * dist
            px = cx - dist * Marker.JUMP_LEFT_ANGLE_COS
            py = cy - dist * Marker.JUMP_LEFT_ANGLE_SIN

        # obs_angle = math.acos()

        px, py = (int(px), int(py))

        # adjust pos

        mask = np.zeros((h + 2, w + 2), np.uint8)

        if px < 0 or px >= w or py < 0 or py >= h:
            return px, py
        
        _, _, _, (x, y, w, h) = \
            cv2.floodFill(screen, mask, (px, py),
                          (0, 0, 0), (4, 4, 4), (4, 4, 4),
                          cv2.FLOODFILL_MASK_ONLY)

        return int(x + w / 2), int(y + h / 2)

    def mark(self, screen):
        bottle_pos, _ = self.find_bottle(screen)
        next = self.next(screen, bottle_pos)

        Util.pin(screen, next)

        return bottle_pos, next

class Muscle:
    VZ_DUR_RATIO = 70 # vel_z / duration
    VY_DUR_RATIO = 15
    VY_DUR_OFS = 135 # vel_y = duration * VY_DUR_RATIO + VY_DUR_OFS
    GRAVITY = 720

    def __init__(self):
        pass

    def duration(self, cur, next):
        bx, by = cur
        nx, ny = next

        dist = ((bx - nx) ** 2 + (by - ny) ** 2) ** 0.5

        dur = 1.31 * dist / Device.RESIZE + 107
        return dur

        dist /= 6.6 # Marker.UNIT # convert to TJS unit

        # print(Marker.UNIT)

        # (VY_DUR_RATIO * dur + 135) / GRAVITY * 2 * VZ_DUR_RATIO * dur == dist
        # k = 1 / GRAVITY * 2 * VZ_DUR_RATIO
        # (VY_DUR_RATIO * dur + 135) * k * dur == dist
        # VY_DUR_RATIO * dur ** 2 * k + 135 * k * dur == dist
        # a * dur ** 2 + b * dur == dist

        k = 1 / Muscle.GRAVITY * 2 * Muscle.VZ_DUR_RATIO
        a = Muscle.VY_DUR_RATIO * k
        b = 135 * k
        c = -dist

        delta = b ** 2 - 4 * a * c
        assert delta >= 0
        dur = (-b + delta ** 0.5) / a / 2 * 1000

        print(dist, k, a, b, dur)

        return dur

dev = Device()
marker = Marker("bottle.png")
muscle = Muscle()

screen = dev.screencap()
marker.calib(screen) # 2.820690

res = marker.mark(screen)

# cv2.imshow("screen", screen)
# cv2.waitKey(0)

last_dur = -INF
mode = "coach"

while True:
    screen = dev.screencap()
    res = marker.mark(screen)
    dur = muscle.duration(*res)

    if (last_dur == dur and mode == "auto") or mode == "jump":
        dev.press(dur)

        if mode == "jump": mode = "coach"
    else:
        last_dur = dur

    cv2.imshow("screen", screen)

    key = cv2.waitKey(1) & 0xff

    if key == ord("c"):
        mode = "auto"
        print("op#auto mode")
    elif key == ord("s"):
        mode = "coach"
        print("op#coach mode")
    elif key == ord("j"):
        mode = "jump"
        print("op#jump")
