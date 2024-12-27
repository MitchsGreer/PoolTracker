"""Table utilites."""
import math
from typing import List, Tuple, Union, Optional
from pathlib import Path
from math import atan2, pi

import matplotlib.pyplot as plt
import cv2
import numpy as np
from numpy.typing import NDArray

from pool_index.ball import PoolBall
from pool_index.util import BallColors, FileT, DebugT


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
        # hsv = cv2.cvtColor(hsv, cv2.COLOR_RGB2HSV)

        hist = cv2.calcHist([hsv], [0], None, [256], [0, 256])
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
            cv2.drawContours(output, [box], 0, (0, 0, 0), 2)

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
            OpenCVTable._euclidian_distance(rgb, BallColors[color].value)
            for color in BallColors._member_names_
        ]
        min_distance = min(distances)
        min_index = distances.index(min_distance)

        return BallColors[BallColors._member_names_[min_index]].name

    @staticmethod
    def find_close_contours(ctrs: List[NDArray], distance_tresh: float) -> List[NDArray]:
        """Find contrours that are within some distance threshold.

        Args:
            ctrs: THe contours to find close ones for.
            distance_tresh: The distance threshold for 'close'.

        Returns:
            A list of close thresholds.
        """
        close_contours = []
        checked = set()
        for i, contour in enumerate(ctrs):
            bbox = cv2.boundingRect(contour)

            for j, c_contour in enumerate(ctrs):
                c_bbox = cv2.boundingRect(c_contour)

                # Don't compare with ourself.
                if bbox == c_bbox:
                    continue

                x = bbox[0]
                y = bbox[1]

                c_x = c_bbox[0]
                c_y = c_bbox[1]

                if OpenCVTable._euclidian_distance([x, y, 0], [c_x, c_y, 0]) < distance_tresh:
                    checked_indicies = tuple(sorted([i, j]))

                    if checked_indicies not in checked:
                        close_contours.append((checked_indicies, (contour, c_contour)))
                        checked.add(checked_indicies)

        return close_contours

    @staticmethod
    def _merge_contours(ctrs: List[NDArray]) -> NDArray:
        """Merge the given contours into one contour.

        Lifted from: https://stackoverflow.com/questions/44501723/how-to-merge-contours-in-opencv

        Args:
            ctrs: The contours to merge.

        Returns:
            The merged countour.
        """
        # Get all the points into the same place.
        list_of_pts = []
        for ctr in ctrs:
            list_of_pts += [pt[0] for pt in ctr]

        # Sort the points in some kind of order.
        center_pt = np.array(list_of_pts).mean(axis = 0) # get origin
        clock_ang_dist = clockwise_angle_and_distance(center_pt) # set origin
        list_of_pts = sorted(list_of_pts, key=clock_ang_dist) # use to sort

        # Force the list into cv2 format.
        ctr = np.array(list_of_pts).reshape((-1,1,2)).astype(np.int32)

        # Make a shape out of the points.
        return cv2.convexHull(ctr)

    @staticmethod
    def _merge_close_contours(ctrs: List[NDArray], distance_tresh: float) -> List[NDArray]:
        """Merge close contours.

        Args:
            ctrs: The contours to merge.
            distance_tresh: The distance threshold for 'close'.

        Returns:
            The merged contour list.
        """
        contours = OpenCVTable.find_close_contours(ctrs, distance_tresh)
        if contours:
            merged_contours = []
            for contour in contours:
                merged_contours.append(OpenCVTable._merge_contours(contour[1]))

            indexes_to_removed = set()
            for contour in contours:
                indexs = contour[0]
                indexes_to_removed.add(indexs[0])
                indexes_to_removed.add(indexs[1])
            indexes_to_removed = list(indexes_to_removed)

            ctrs = list(ctrs)
            for index in indexes_to_removed:
                ctrs.pop(index)

                for i in range(len(indexes_to_removed)):
                    indexes_to_removed[i] -= 1

            ctrs = list(ctrs) + merged_contours

        return ctrs

    @staticmethod
    def _euclidian_distance(
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

    def detect_balls(self, debug: bool = DebugT, debug_folder: Optional[FileT] = None) -> None:
        """Detect pool balls on this table.

        TODO: Add a debug folder to save images to.
        TODO: Add debugging levels to allow for just the filtered objects to be
              saved.

        Args:
            debug: Level of debug for this method.
            debug_folder: The folder to save debug images to.
        """
        transformed = self.image.copy()

        transformed_blur = cv2.GaussianBlur(transformed, (0, 0), 2)  # blur applied
        blur_RGB = cv2.cvtColor(transformed_blur, cv2.COLOR_BGR2RGB)  # rgb version

        # hsv colors of the snooker table
        hsv = cv2.cvtColor(blur_RGB, cv2.COLOR_RGB2HSV)  # convert to hsv
        if debug == DebugT.LEVEL_2:
            cv2.imwrite(str(Path(debug_folder, "hsv.png")), hsv)

        self.lower, self.upper = self._felt_color()
        mask = cv2.inRange(transformed, self.lower, self.upper)  # table's mask

        # apply closing
        kernel = np.ones((5, 5), np.uint8)
        mask_closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # dilate->erode

        # Invert mask to focus on objects on table.
        _, mask_inv = cv2.threshold(mask_closing, 5, 255, cv2.THRESH_BINARY_INV)

        if debug == DebugT.LEVEL_2:
            # masked image with inverted mask
            masked_img = cv2.bitwise_and(transformed, transformed, mask=mask_inv)

            cv2.imwrite(str(Path(debug_folder, "transformed_blur.png")), transformed_blur)
            cv2.imwrite(str(Path(debug_folder, "mask_closing.png")), mask_closing)
            cv2.imwrite(str(Path(debug_folder, "masked_img.png")), masked_img)

        # Find contours and filter them.
        ctrs, _ = cv2.findContours(
            mask_inv, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )

        # Merge contours so we can get balls cut in half.
        ctrs = self._merge_close_contours(ctrs, 10)

        if debug == DebugT.LEVEL_2:
            detected_objects = self._draw_rectangles(ctrs, transformed)
            cv2.imwrite(str(Path(debug_folder, "detected_objects.png")), detected_objects)

        ctrs_filtered = self._filter_ctrs(
            ctrs
        )  # filter unwanted contours (wrong size or shape)

        if debug == DebugT.LEVEL_2:
            # draw contours after filter
            detected_objects_filtered = self._draw_rectangles(
                ctrs_filtered, transformed
            )  # filtered detected objects will be marked in boxes

            cv2.imwrite(
                str(Path(debug_folder, "detected_objects_filtered.png")), detected_objects_filtered
            )

        # find average color inside contours:
        ctrs_color, colors = self._find_ctrs_color(ctrs_filtered, transformed)

        if debug == DebugT.LEVEL_2:
            # contours color image + transformed image
            ctrs_color = cv2.addWeighted(ctrs_color, 0.5, transformed, 0.5, 0)

            cv2.imwrite(str(Path(debug_folder, "ctrs_color.png")), ctrs_color)

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

class clockwise_angle_and_distance():
    '''
    A class to tell if point is clockwise from origin or not.
    This helps if one wants to use sorted() on a list of points.

    Parameters
    ----------
    point : ndarray or list, like [x, y]. The point "to where" we g0
    self.origin : ndarray or list, like [x, y]. The center around which we go
    refvec : ndarray or list, like [x, y]. The direction of reference

    use:
        instantiate with an origin, then call the instance during sort
    reference:
    https://stackoverflow.com/questions/41855695/sorting-list-of-two-dimensional-coordinates-by-clockwise-angle-using-python

    Returns
    -------
    angle

    distance


    '''
    def __init__(self, origin):
        self.origin = origin

    def __call__(self, point, refvec = [0, 1]):
        if self.origin is None:
            raise NameError("clockwise sorting needs an origin. Please set origin.")
        # Vector between point and the origin: v = p - o
        vector = [point[0]-self.origin[0], point[1]-self.origin[1]]
        # Length of vector: ||v||
        lenvector = np.linalg.norm(vector[0] - vector[1])
        # If length is zero there is no angle
        if lenvector == 0:
            return -pi, 0
        # Normalize vector: v/||v||
        normalized = [vector[0]/lenvector, vector[1]/lenvector]
        dotprod  = normalized[0]*refvec[0] + normalized[1]*refvec[1] # x1*x2 + y1*y2
        diffprod = refvec[1]*normalized[0] - refvec[0]*normalized[1] # x1*y2 - y1*x2
        angle = atan2(diffprod, dotprod)
        # Negative angles represent counter-clockwise angles so we need to
        # subtract them from 2*pi (360 degrees)
        if angle < 0:
            return 2*pi+angle, lenvector
        # I return first the angle because that's the primary sorting criterium
        # but if two vectors have the same angle then the shorter distance
        # should come first.
        return angle, lenvector
