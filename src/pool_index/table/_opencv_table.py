"""Table utilites."""
import math
from typing import List, Tuple, Union

import cv2
import numpy as np
from numpy.typing import NDArray

from pool_index.ball import PoolBall
from pool_index.util import BallColors, FileT


class OpenCVTable:
    """Utilities for the table."""

    def __init__(self, img: Union[FileT, NDArray]) -> None:
        """Constructor for Table."""

        try:
            self.image = cv2.imread(img)
        except:
            self.image = img

        self.detected_balls: List[PoolBall] = []

    @staticmethod
    def _get_index_of_min(Data_List: List[int]) -> List[int]:
        """
        Return as list of the indexes of the minmum values in a 1D array of data

        Args:
            Data_List: List of HSV values.
        Returns:
            The list of indicies of the min HSV values in the given list.
        """
        # make sure data is in a standard list, not a numpy array
        if type(Data_List).__module__ == np.__name__:
            Data_List = list(Data_List)

        # return a list of the indexes of the minimum values. Important if there is >1 minimum
        return [i for i, x in enumerate(Data_List) if x == min(Data_List)]

    @staticmethod
    def _get_index_of_max(Data_List: List[int]) -> List[int]:
        """Return as list of the indexes of the maximum values in a 1D array of data.

        Args:
            Data_List: List of HSV values.
        Returns:
            The list of indicies of the max HSV values in the given list.
        """
        # make sure data is in a standard list, not a numpy array
        if type(Data_List).__module__ == np.__name__:
            Data_List = list(Data_List)

        # return a list of the indexes of the max values. Important if there is >1 maximum
        return [i for i, x in enumerate(Data_List) if x == max(Data_List)]

    def _felt_color(self, search_width: float = 45) -> Tuple[NDArray, NDArray]:
        """Find the most common HSV values in the image.

        In a well lit image, this will be the cloth

        Args:
            search_width: The width of the search in the image foe the high and low. Defaults to 45.

        Returns:
            The high and low HSV values in the given image.
        """
        hsv = self.image.copy()
        hsv = cv2.cvtColor(hsv, cv2.COLOR_RGB2HSV)

        hist = cv2.calcHist([hsv], [1], None, [180], [0, 180])
        h_max = self._get_index_of_max(hist)[0]

        hist = cv2.calcHist([hsv], [1], None, [256], [0, 256])
        s_max = self._get_index_of_max(hist)[0]

        hist = cv2.calcHist([hsv], [2], None, [256], [0, 256])
        v_max = self._get_index_of_max(hist)[0]

        # define range of blue color in HSV
        lower_color = np.array(
            [h_max - search_width, s_max - search_width, v_max - search_width]
        )
        upper_color = np.array(
            [h_max + search_width, s_max + search_width, v_max + search_width]
        )

        return lower_color, upper_color

    @staticmethod
    def _filter_ctrs(
        ctrs: NDArray,
        min_s: float = 90,
        max_s: float = 600,
        alpha: float = 5,
        square_thresh: float = 5,
    ) -> List[NDArray]:
        """Filter contours, exlude any of them that don't look like pool balls.

        Args:
            ctrs: Contours to filter.
            min_s: The min size of the pool ball. Defaults to 90.
            max_s: The max size of the pool ball. Defaults to 600.
            alpha: The percent difference between the height and width. Defaults to 5.
            square_thresh: The hard difference between height and width. Defaults to 5.

        Returns:
            The list of filtered contours.
        """
        filtered_ctrs: List[NDArray] = []  # list for filtered contours

        for x in range(len(ctrs)):  # for all contours
            rot_rect = cv2.minAreaRect(ctrs[x])  # area of rectangle around contour
            width = rot_rect[1][0]  # width of rectangle
            heigth = rot_rect[1][1]  # height
            area = cv2.contourArea(ctrs[x])  # contour area

            # If the contour isn't the size of a snooker ball
            if (heigth * alpha < width) or (width * alpha < heigth):
                continue  # do nothing

            # If the contour area is too big/small.
            if (area < min_s) or (area > max_s):
                continue  # do nothing

            # If the sides of the box are not square enough.
            if abs(width - heigth) > square_thresh:
                continue  # do nothing

            # If it failed previous statements then it is most likely a ball.
            filtered_ctrs.append(ctrs[x])

        return filtered_ctrs  # returns filtere contours

    @staticmethod
    def _draw_rectangles(ctrs: List[NDArray], img: NDArray) -> NDArray:
        """Draw rectangle around the contours on the given image,

        Args:
            ctrs: The list of contours to trace.
            img: The image to draw the rectangles on.

        Returns:
            The image with the rectangles drawn on.
        """
        output = img.copy()

        for i in range(len(ctrs)):
            rot_rect = cv2.minAreaRect(ctrs[i])
            box = np.int64(cv2.boxPoints(rot_rect))
            cv2.drawContours(output, [box], 0, (255, 100, 0), 2)

        return output

    @staticmethod
    def _find_ctrs_color(
        ctrs: List[NDArray], input_img: NDArray
    ) -> Tuple[NDArray, List[Tuple[int, int, int]]]:
        """Find and draw on the colors of the contours.

        Args:
            ctrs: The contours to draw collered balls for.
            input_img: The image to draw the balls on.

        Returns:
            The image wirh all contours drawn on and a list of their colors.
        """
        K = np.ones((3, 3), np.uint8)  # filter
        output = input_img.copy()  # np.zeros(input_img.shape,np.uint8) # empty img
        gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)  # gray version
        mask = np.zeros(gray.shape, np.uint8)  # empty mask
        colors = []

        for i in range(len(ctrs)):  # for all contours
            # find center of contour
            M = cv2.moments(ctrs[i])
            cX = int(M["m10"] / M["m00"])  # X pos of contour center
            cY = int(M["m01"] / M["m00"])  # Y pos

            mask[...] = 0  # reset the mask for every ball

            # draws the mask of current contour (every ball is getting masked each iteration)
            cv2.drawContours(mask, ctrs, i, 255, -1)

            # erode mask to filter green color around the balls contours
            mask = cv2.erode(mask, K, iterations=3)

            color = cv2.mean(input_img, mask)

            # Color is (B, G, R, ?), grab the RGB value.
            rgb = [int(c) for c in color[:3]]
            rgb.reverse()
            colors.append(rgb)

            output = cv2.circle(
                output,  # img to draw on
                (cX, cY),  # position on img
                20,  # radius of circle - size of drawn snooker ball
                color,  # color mean of each contour-color of each ball (src_img=transformed img)
                -1,  # -1 to fill ball with color
            )

        return output, colors

    @staticmethod
    def _find_color(rgb: Tuple[int, int, int]) -> str:
        """Find the closet pool ball color for the given RGB values.

        Args:
            rgb: The RGB values to check.

        Returns:
            The string value of the color.
        """

        distances = [
            OpenCVTable._euclidian_disance(rgb, BallColors[color].value)
            for color in BallColors._member_names_
        ]
        min_distance = min(distances)
        min_index = distances.index(min_distance)

        return BallColors[BallColors._member_names_[min_index]].name

    @staticmethod
    def _euclidian_disance(
        point_1: Tuple[int, int, int], point_2: Tuple[int, int, int]
    ) -> float:
        """Distance between two points in three-dimensional space.

        d = √ [(x2 - x1)2 + (y2 - y1)2 + (z2 - z1)2].

        Args:
            point_1: A point in three-dimensional space.
            point_2: A point in three-dimensional space.

        Returns:
            The distance between the points.
        """
        return math.sqrt(
            (point_1[0] - point_2[0]) ** 2
            + (point_1[1] - point_2[1]) ** 2
            + (point_1[2] - point_2[2]) ** 2
        )

    def detect_balls(self, debug: bool = False) -> None:
        """Detect pool balls on this table.

        TODO: Add a debug folder to save images to.
        TODO: Add debugging levels to allow for just the filtered objects to be
              saved.

        """
        transformed = self.image.copy()

        transformed_blur = cv2.GaussianBlur(transformed, (0, 0), 2)  # blur applied
        blur_RGB = cv2.cvtColor(transformed_blur, cv2.COLOR_BGR2RGB)  # rgb version

        # hsv colors of the snooker table
        hsv = cv2.cvtColor(blur_RGB, cv2.COLOR_RGB2HSV)  # convert to hsv
        lower, upper = self._felt_color()
        mask = cv2.inRange(hsv, lower, upper)  # table's mask

        # apply closing
        kernel = np.ones((5, 5), np.uint8)
        mask_closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # dilate->erode

        # Invert mask to focus on objects on table.
        _, mask_inv = cv2.threshold(mask_closing, 5, 255, cv2.THRESH_BINARY_INV)

        if debug:
            # masked image with inverted mask
            masked_img = cv2.bitwise_and(transformed, transformed, mask=mask_inv)

            cv2.imwrite("images/gen/transformed_blur.png", transformed_blur)
            cv2.imwrite("images/gen/mask_closing.png", mask_closing)
            cv2.imwrite("images/gen/masked_img.png", masked_img)

        # Find contours and filter them.
        ctrs, hierarchy = cv2.findContours(
            mask_inv, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )

        if debug:
            detected_objects = self._draw_rectangles(ctrs, transformed)
            cv2.imwrite("images/gen/detected_objects.png", detected_objects)

        ctrs_filtered = self._filter_ctrs(
            ctrs
        )  # filter unwanted contours (wrong size or shape)

        if debug:
            # draw contours after filter
            detected_objects_filtered = self._draw_rectangles(
                ctrs_filtered, transformed
            )  # filtered detected objects will be marked in boxes

            cv2.imwrite(
                "images/gen/detected_objects_filtered.png", detected_objects_filtered
            )

        # find average color inside contours:
        ctrs_color, colors = self._find_ctrs_color(ctrs_filtered, transformed)

        if debug:
            # contours color image + transformed image
            ctrs_color = cv2.addWeighted(ctrs_color, 0.5, transformed, 0.5, 0)

            cv2.imwrite("images/gen/ctrs_color.png", ctrs_color)

        for ball, color in zip(ctrs_filtered, colors):
            bbox = cv2.boundingRect(ball)
            color_name = self._find_color(color)
            self.detected_balls.append(PoolBall(bbox, color, color_name))

    def balls_up(self) -> List[str]:
        """Return the balls that are still up on the table.

        Returns:
            The list of balls that should be still on the table.
        """
        ball_map = {
            "YELLOW": 1,
            "BLUE": 2,
            "RED": 3,
            "PURPLE": 4,
            "ORANGE": 5,
            "GREEN": 6,
            "MAROON": 7,
            "BLACK": 8,
            "PINK": 4,
            "WHITE": "Cue",
        }

        balls_up = []
        for ball in self.detected_balls:
            balls_up.append(ball_map[ball.number])

        return balls_up
