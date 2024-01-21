Pool Index
==========
This application detects balls on a pool table. This project aims to locate pool
balls in a couple of ways.

OpenCV
------
An OpenCV method has been implemented with the following success and failures:

Success:
- Most balls are found.
- Most colors are recognized correctly.

Failures:
- The table must be in an eagle eye view.
    - This can be addressed by stretching the image to the desired format.
- Phantom balls are detected outside the pool table.
    - This could be solved by bounding the pool table, there is less noise on
    the felt of the table.
- It is hard to discern stripped verses solid balls.
    - This can be solved by looking at the amount of black or white pixels in
    the detected ball.
- It is hard to see balls when they are clumped together.

Next Steps:
- Use the bounsing boxes of the balls to create training datasets for the object
detection part of this repository.

This project is built of a mix of th following git repositories:
- https://github.com/sgrieve/PoolTable
- https://github.com/Dimnir/TrackingSnookerBalls

Object Detection
----------------
The object detection method will be implemented usng

- https://github.com/WongKinYiu/yolov7
- https://github.com/opencv/cvat
