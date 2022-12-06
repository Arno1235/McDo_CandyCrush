import pretty_errors

import numpy as np
from PIL import Image, ImageGrab
import cv2

import copy

import pyautogui
import time


def save_element(main_screen, element_width, element_height):

    x, y = 3, 3

    el = get_element(main_screen, element_width, element_height, x, y)

    cv2.imwrite("el5.png", el)

    import sys
    sys.exit()


def get_element(main_screen, element_width, element_height, x, y):
    el = main_screen[y * element_height:(y + 1) * element_height,
                     x * element_width:(x + 1) * element_width]
    # cv2.imshow("frame", el)
    # cv2.waitKey(0)
    return el


class Image_Matcher:

    def __init__(self):

        images = ["./elements/el1.png", "./elements/el2.png",
                  "./elements/el3.png", "./elements/el4.png", "./elements/el5.png"]
        self.all_features = np.zeros(shape=(len(images), 3))
        # self.star_features = np.zeros(shape=(len(images)+1, 3))

        for i in range(len(images)):
            feature = self.extract(img=cv2.imread(images[i]))
            self.all_features[i] = np.array(feature)

        # for i in range(len(images)):
        #     self.match_star(img=cv2.imread(images[i]))

        # star = "./elements/el6.png"

        # self.match_star(img=cv2.imread(star))

        # import sys
        # sys.exit(0)

    def extract(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        lower_red = (0, 50, 50)
        upper_red = (10, 255, 255)

        lower_yellow = (20, 0, 100)
        upper_yellow = (40, 255, 255)

        lower_white = (0, 0, 100)
        upper_white = (255, 50, 255)

        mask_red = cv2.inRange(hsv, lower_red, upper_red)
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        mask_white = cv2.inRange(hsv, lower_white, upper_white)

        return [np.count_nonzero(mask_red), np.count_nonzero(mask_yellow), np.count_nonzero(mask_white)]

    def extract_green(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        lower_green = (30, 0, 0)
        upper_green = (120, 255, 255)

        mask_green = cv2.inRange(hsv, lower_green, upper_green)

        cv2.imshow("test", mask_green)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        ret = np.count_nonzero(mask_green) / \
            (mask_green.shape[0] * mask_green.shape[1])

        print(ret)

        return ret

    def match_star(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        lower_green = (30, 30, 0)
        upper_green = (120, 255, 255)

        mask_green = cv2.inRange(hsv, lower_green, upper_green)

        # cv2.imshow("test", mask_green)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        ret, thresh = cv2.threshold(mask_green, 127, 255, 1)

        contours, h = cv2.findContours(thresh, 1, 2)

        for cnt in contours:
            approx = cv2.approxPolyDP(cnt, 0.02*cv2.arcLength(cnt, True), True)
            # print(len(approx))

        circles = cv2.HoughCircles(
            mask_green, cv2.HOUGH_GRADIENT, 1, 100, param1=30, param2=10, minRadius=20, maxRadius=120)

        # if circles is None:
        #     print("None")
        # else:
        #     print(len(circles))

        # if circles is None:
        #     print(len(contours))
        #     print(len(approx))
        #     print(circles is None)

        # print(len(contours) == 1 and len(approx) == 10 and circles is None)

        return len(contours) == 1 and len(approx) == 10 and circles is None
        # return circles is None

    def match(self, img):

        if self.match_star(img):
            return 10

        # Match image
        query = self.extract(img=img)  # Extract its features
        # Calculate the similarity (distance) between images
        dists = np.linalg.norm(self.all_features - query, axis=1)
        # Extract 5 images that have lowest distance
        ids = np.argsort(dists)[0]
        return ids


def load_grid(main_screen, grid_size, element_width, element_height):

    grid = []

    for y in range(grid_size):

        row = []
        for x in range(grid_size):
            el = get_element(main_screen, element_width, element_height, x, y)
            tile = imgMatch.match(el)
            row.append(tile)

            # cv2.imshow(str(tile), el)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

        grid.append(row)

    return grid

# TODO: Check for star if other star is in the same file
# TODO: Before star check to create other star (should know amount of moves left)
# TODO: Move star towards middle


def calculate_best_move(grid):

    best_x, best_y = -1, -1
    best_vert = -1
    best_score = 0

    for y in range(len(grid)):
        for x in range(len(grid)):
            if grid[y][x] == 10:
                if x == len(grid) - 1:
                    x -= 1
                best_x, best_y = x, y
                best_vert = 0
                best_score = 6

    for y in range(len(grid)):
        for x in range(len(grid)-1):
            tmp_grid = copy.deepcopy(grid)

            tmp_grid = do_move_horizontal(tmp_grid, x, y)

            score, tmp_grid = calculate_score(tmp_grid)

            if score > best_score:
                best_score = score
                best_x = x
                best_y = y
                best_vert = 0

    for y in range(len(grid)-1):
        for x in range(len(grid)):
            tmp_grid = copy.deepcopy(grid)

            tmp_grid = do_move_vertical(tmp_grid, x, y)

            score, tmp_grid = calculate_score(tmp_grid)

            if score > best_score:

                best_score = score
                best_x = x
                best_y = y
                best_vert = 1

    return best_x, best_y, best_vert, best_score


def do_move_horizontal(grid, x, y):

    grid[y][x], grid[y][x+1] = grid[y][x+1], grid[y][x]
    return grid


def do_move_vertical(grid, x, y):

    grid[y][x], grid[y+1][x] = grid[y+1][x], grid[y][x]
    return grid


def calculate_score(grid):

    score = 0

    improved = True

    while improved:
        improved = False

        el_to_clear = []

        for l in range(3, len(grid)):

            for y in range(len(grid)):
                for x in range(len(grid)-l+1):
                    if grid[y][x] == -1:
                        continue

                    if all(grid[y][i] == grid[y][x] for i in range(x, x+l)):
                        score += l
                        for i in range(x, x+l):
                            el_to_clear.append((i, y))
                        improved = True

            for x in range(len(grid)):
                for y in range(len(grid)-l):
                    if grid[y][x] == -1:
                        continue

                    if all(grid[i][x] == grid[y][x] for i in range(y, y+l)):
                        score += l
                        for i in range(y, y+l):
                            el_to_clear.append((x, i))
                        improved = True

        if improved:
            for x, y in el_to_clear:
                grid[y][x] = -1

            grid = do_cascade(grid)

    return score, grid


def do_cascade(grid):

    improved = True

    while improved:
        improved = False

        for x in range(len(grid)):
            for y in range(len(grid)-1, 0, -1):
                if grid[y][x] == -1 and grid[y-1][x] != -1:
                    grid = do_move_vertical(grid, x, y-1)
                    improved = True

    return grid


def print_grid(grid):
    print("================================")
    for row in grid:
        print(row)
    print("================================")


def take_screenshot(left, top, width, height):
    img = ImageGrab.grab(bbox=(left, top, width, height))  # x, y, w, h
    img_np = np.array(img)
    frame = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    # cv2.imshow("frame", frame)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return frame


def make_move(x, y, vertical, left, top, element_width, element_height):

    pyautogui.moveTo((element_width/2 + left + x * element_width)/2,
                     (element_height/2 + top + y * element_height)/2)

    if vertical:
        pyautogui.drag(0, element_height*0.6, 0.2, button='left')
    else:
        pyautogui.drag(element_width*0.6, 0, 0.2, button='left')

    time.sleep(1)

    pyautogui.moveTo(500, 500)


def check_next_level(main_screen):

    test_strip = main_screen[-10:-1, 0:-1]
    hsv = cv2.cvtColor(test_strip, cv2.COLOR_BGR2HSV)

    lower_yellow = (20, 0, 100)
    upper_yellow = (40, 255, 255)

    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    return np.count_nonzero(mask_yellow) / (mask_yellow.shape[0] * mask_yellow.shape[1]) > 0.8


def go_to_next_level():

    pyautogui.moveTo((left + 3 * element_width)/2,
                     (top + 6.5 * element_height)/2)

    pyautogui.click()

    time.sleep(1)

    pyautogui.moveTo(500, 500)

    time.sleep(7)


if __name__ == "__main__":

    imgMatch = Image_Matcher()

    # im = cv2.imread('test.png')

    # element_width = 206
    # left, top = 100, 900

    left, top = 65, 670
    width, height = 830, 1450

    grid_size = 6

    element_width = round((width - left) / grid_size)
    element_height = round((height - top) / grid_size)

    # main_screen = take_screenshot(left, top, width, height)

    # save_element(main_screen, element_width, element_height)

    # grid = load_grid(main_screen, grid_size, element_width, element_height)

    # print_grid(grid)

    # x, y, vertical, best_score = calculate_best_move(grid)

    # print(x, y, vertical)

    # make_move(x, y, vertical, left, top, element_width, element_height)

    while True:

        main_screen = take_screenshot(left, top, width, height)

        if check_next_level(main_screen):
            # break
            go_to_next_level()
            main_screen = take_screenshot(left, top, width, height)

        grid = load_grid(main_screen, grid_size, element_width, element_height)

        print_grid(grid)

        x, y, vertical, best_score = calculate_best_move(grid)

        print(x, y, vertical, best_score)

        if best_score < 160:

            make_move(x, y, vertical, left, top, element_width, element_height)

            time.sleep(5)

# 3579
