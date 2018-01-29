#! /usr/bin/python3

import random
import numpy as np
from PIL import Image
import math
import time
import sys
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

    @staticmethod
    def dist(apos, bpos):
        return ((apos[0] - bpos[0]) ** 2 + (apos[1] - bpos[1]) ** 2) ** 0.5

class Device:
    PRESS_POINT = (600, 600)
    RESIZE = 0.3 # resize the input image for better performance

    def __init__(self):
        h, w = self.screencap().shape[:2]
        Device.PRESS_POINT = (w / 2 / Device.RESIZE, h / 2 / Device.RESIZE)

    def screencap(self):
        raw = self.screenraw()
        img = cv2.imdecode(np.fromstring(raw, np.uint8), cv2.IMREAD_COLOR)

        img = Util.resize(img, Device.RESIZE)

        return img

    def press(self, duration):
        self.taphold(*Device.PRESS_POINT, duration)

class AndroidDevice(Device):
    def __init__(self):
        import adb as pyadb3

        self.adb = pyadb3.ADB()

        super(AndroidDevice, self).__init__()

    def screenraw(self):
        self.adb.shell_command("screencap /sdcard/bottle-test.png")
        self.adb.run_cmd([ "pull", "/sdcard/bottle-test.png", "bottle-test.png" ])

        with open("bottle-test.png", "rb") as fp:
            cont = fp.read()

        return cont

    def taphold(self, x, y, duration):
        cmd = "input swipe %d %d %d %d %d" % (x, y, x, y, duration)
        print(cmd)
        self.adb.shell_command(cmd)

class iOSDevice(Device):
    def __init__(self):
        import wda

        self.client = wda.Client()
        self.session = self.client.session()

        super(iOSDevice, self).__init__()

    def screenraw(self):
        return self.client.screenshot()

    def taphold(self, x, y, duration):
        self.session.taphold(x, y, duration)

class Measure:
    PIVOT_POS = None
    UNIT = -1 # pixel / TJSU
    SCALE = 0

# image -> start & end point
class Marker:
    PIVOT_POS = None
    INIT_BLOCK_DIST = 20 # distance in TJSU

    CENTER_DELTA = (45, 35) # in pixels
    BOTTLE_HEIGHT = 5.76 # constant in the game = head_pos + head_radius
    UNIT = -1 # pixel / TJSU

    JUMP_RIGHT_ANGLE_TAN = 0.5762124386941568 # see notes # 0.579
    JUMP_LEFT_ANGLE_TAN = 0.5419172526716592 # 0.555

    JUMP_RIGHT_ANGLE_SIN = JUMP_RIGHT_ANGLE_TAN / ((1 + JUMP_RIGHT_ANGLE_TAN ** 2) ** 0.5)
    JUMP_LEFT_ANGLE_SIN = JUMP_LEFT_ANGLE_TAN / ((1 + JUMP_LEFT_ANGLE_TAN ** 2) ** 0.5)

    JUMP_RIGHT_ANGLE_COS = JUMP_RIGHT_ANGLE_SIN / JUMP_RIGHT_ANGLE_TAN
    JUMP_LEFT_ANGLE_COS = JUMP_LEFT_ANGLE_SIN / JUMP_LEFT_ANGLE_TAN

    JUMP_RIGHT_ANGLE = math.asin(JUMP_RIGHT_ANGLE_SIN)
    JUMP_LEFT_ANGLE = math.asin(JUMP_LEFT_ANGLE_SIN)

    # x axis is towards right
    X_AXIS_SCREEN_ANGLE = 0.8224182792713783
    Z_AXIS_SCREEN_ANGLE = 0.7483780475235182

    def __init__(self, path):
        self.bottle = cv2.imread(path, 0)
        self.prev_dir = 1

    def find_bottle(self, screen):
        h, w = self.bottle.shape

        res = cv2.matchTemplate(cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY),
                                self.bottle, cv2.TM_CCOEFF_NORMED)

        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        pos = (int(max_loc[0] + w / 2), int(max_loc[1] + h * 0.9))

        cv2.rectangle(screen, max_loc, (max_loc[0] + w, max_loc[1] + h), (0, 0, 0), 1)
        # cv2.rectangle(screen, min_loc, (min_loc[0] + w, min_loc[1] + h), (0, 0, 255), 2)
        # Util.pin(screen, pos)
        # cv2.imshow("screen", screen)

        # pos -> center of the bottle
        return pos, max_val

    def apply_calib(self):
        self.bottle = Util.resize(self.bottle, 1 / Measure.SCALE)

    def save_calib(self):
        return """class Measure:
    UNIT = %f
    PIVOT_POS = %s
    SCALE = %f""" % (Measure.UNIT, Measure.PIVOT_POS, Measure.SCALE)

    # calib :: screen -> update self.bottle Measure.UNIT
    # ASSERT: screen is at the initial position
    def calib(self, screen):
        max_val = -INF
        max_scale = 0

        for scale in np.linspace(0.2, 4, 30).tolist():
            print("trying scale %f" % scale)

            nscreen = Util.resize(screen, scale)   
            _, val = self.find_bottle(nscreen)

            if val > max_val:
                max_val = val
                max_scale = scale
        
        print("optimal scale %f" % max_scale)

        self.bottle = Util.resize(self.bottle, 1 / max_scale)

        Measure.SCALE = max_scale

        # estimated unit
        # Measure.UNIT = self.bottle.shape[0] * 0.9 / Marker.BOTTLE_HEIGHT

        h, w = screen.shape[:2]

        # estimated pivot
        Measure.PIVOT_POS = \
            int((w + Marker.CENTER_DELTA[0]) / 2), \
            int((h + Marker.CENTER_DELTA[1]) / 2)

        bottle_pos, next_pos, *_ = self.mark(screen)

        print(bottle_pos, next_pos)

        dist = Util.dist(bottle_pos, next_pos)
        dist = Marker.rd_x(dist)

        Measure.UNIT = dist / Marker.INIT_BLOCK_DIST

        Measure.PIVOT_POS = \
            int((next_pos[0] + bottle_pos[0]) / 2), \
            int((next_pos[1] + bottle_pos[1]) / 2)

        print(self.bottle.shape[0] / Measure.UNIT)

        return Measure

    # real distance in x axis
    @staticmethod
    def rd_x(dist):
        return dist / math.sin(Marker.X_AXIS_SCREEN_ANGLE)
    
    # real distance in z axis
    @staticmethod
    def rd_z(dist):
        return dist / math.sin(Marker.Z_AXIS_SCREEN_ANGLE)

    def next(self, screen, bottle_pos):
        h, w = screen.shape[:2]
        bx, by = bottle_pos

        # if Measure.PIVOT_POS == None:
        #     # use estimated pivot
        #     cx, cy = int((w + Marker.CENTER_DELTA[0]) / 2), \
        #              int((h + Marker.CENTER_DELTA[1]) / 2)
        # else:
        
        cx, cy = Measure.PIVOT_POS

        # cv2.circle(screen, (cx, cy), 4, (100, 100, 100), -1)
        # Util.pin(screen, (cx, cy))

        orig = dist = ((cx - bx) ** 2 + (cy - by) ** 2) ** 0.5

        # correction on turn

        if (cx - bx) * self.prev_dir < 0: # different direction as the previous jump
            if self.prev_dir == 1: # now turn to the left
                a1 = math.atan(abs(cy - by) / abs(cx - bx)) + Marker.JUMP_RIGHT_ANGLE
                turn = -1
            else: # now turn to the right
                a1 = math.atan(abs(cy - by) / abs(cx - bx)) + Marker.JUMP_LEFT_ANGLE
                turn = 1

            d1 = dist * math.sin(a1)
            tot = Marker.JUMP_LEFT_ANGLE + Marker.JUMP_RIGHT_ANGLE

            if tot < math.pi / 2:
                dist = d1 / math.sin(tot)
            else:
                dist = d1 / math.sin(math.pi - tot)
        else:
            turn = 0

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
            return (px, py), self.prev_dir, turn
        else:
            _, _, _, (x, y, w, h) = \
                cv2.floodFill(screen, mask, (px, py),
                            (0, 0, 0), (4, 4, 4), (4, 4, 4),
                            cv2.FLOODFILL_MASK_ONLY)

            return (int(x + w / 2), int(y + h / 2)), self.prev_dir, turn

    def now_center(self, next_pos):
        return 2 * Measure.PIVOT_POS[0] - next_pos[0], \
               2 * Measure.PIVOT_POS[1] - next_pos[1]

    def mark(self, screen):
        bottle_pos, _ = self.find_bottle(screen)
        next, dir, turn = self.next(screen, bottle_pos)
        center = self.now_center(next)
 
        # delta = Util.dist(bottle_pos, center)
        dist = Util.dist(bottle_pos, next)

        # if turn == 1:
        #     delta = Marker.rd_z(delta)
        #     dist = Marker.rd_x(dist)

        #     dist = (delta ** 2 + dist ** 2) ** 0.5
        # elif turn == -1:
        #     delta = Marker.rd_x(delta)
        #     dist = Marker.rd_z(dist)

        #     dist = (delta ** 2 + dist ** 2) ** 0.5
        # else:
        #     if dir == 1:
        #         dist = Marker.rd_x(dist)
        #     else:
        #         dist = Marker.rd_z(dist)

        return bottle_pos, next, dist

    def display(self, screen, bottle_pos, next_pos, *other):
        cv2.line(screen, bottle_pos, next_pos, (0, 255, 0), 1)

        now_center = self.now_center(next_pos)

        cv2.line(screen, now_center, next_pos, (255, 0, 0), 1)

        Util.pin(screen, bottle_pos)
        Util.pin(screen, next_pos)
        Util.pin(screen, now_center)
        Util.pin(screen, Measure.PIVOT_POS, color = (0, 0, 255))

class Muscle:
    VZ_DUR_RATIO = 70 # vel_z / duration
    VY_DUR_RATIO = 15
    VY_DUR_OFS = 135 # vel_y = duration * VY_DUR_RATIO + VY_DUR_OFS
    GRAVITY = 720

    MIN_DELAY = 1.6 # in sec

    def __init__(self):
        pass

    # dist is the distance in 3d in pixels
    def duration(self, cur, next, dist):
        bx, by = cur
        nx, ny = next

        # dist = ((bx - nx) ** 2 + (by - ny) ** 2) ** 0.5

        print(dist)
        
        # dur = 1.3 * dist / Device.RESIZE + 110
        dur = 1.22 * dist / Device.RESIZE + 100
        
        return dur

        # print(dist)
        # dur = 2.8 * (dist / Device.RESIZE) ** 0.9
        # return dur

        dist /= Measure.UNIT # convert to TJS unit
        # dist *= 1.1

        # if next[0] > cur[0]:
        #     # jump right
        #     dist = dist / math.sin(Marker.X_AXIS_SCREEN_ANGLE)
        # else:
        #     # jump left
        #     dist = dist / math.sin(Marker.Z_AXIS_SCREEN_ANGLE)

        # print(Measure.UNIT)

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
        dur = (-b + delta ** 0.5) / (2 * a) * 1000

        print(dist, dur)

        # dur *= 1.2

        return dur

dev = AndroidDevice()
marker = Marker("bottle.png")
muscle = Muscle()

screen = dev.screencap()

if len(sys.argv) >= 2 and sys.argv[1] == "calib":
    marker.calib(screen) # 2.820690

    with open("measure.py", "wb") as fp:
        fp.write(marker.save_calib().encode())
else:
    import measure
    Measure = measure.Measure
    marker.apply_calib() # 2.820690

res = marker.mark(screen)

# cv2.imshow("screen", screen)
# cv2.waitKey(0)

last_dur = -INF
mode = "coach"

last_jump = time.time()

while True:
    screen = dev.screencap()
    res = marker.mark(screen)
    dur = muscle.duration(*res) # + random.uniform(-50, 50)

    marker.display(screen, *res)

    if mode == "auto" or mode == "jump":
        time.sleep(random.uniform(0, 1))
        dev.press(dur)
        jumped = True

        if mode == "jump":
            mode = "coach"
    else:
        last_dur = dur
        jumped = False

    cv2.imshow("screen", screen)

    key = cv2.waitKey(int(Muscle.MIN_DELAY * 1000) if jumped else 1) & 0xff

    if key == ord("c"):
        mode = "auto"
        print("op#auto mode")
    elif key == ord("s"):
        mode = "coach"
        print("op#coach mode")
    elif key == ord("j"):
        mode = "jump"
        print("op#jump")
