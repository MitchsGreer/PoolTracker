"""Create a dataset with youtube data."""
from pathlib import Path
import shutil

from pytube import YouTube
import cv2

from pool_index.util import FileT
from pool_index.table import OpenCVTable

def download_video(url: str, output_dir: str) -> str:
    """Download the video at the given URL.

    Args:
        url: The URL to download the video at.
        output_dir: The directory to save to.

    Returns:
        The path to the file that was saved.
    """
    return YouTube(url).streams.first().download(output_path=output_dir)


def iterate_through_video(video: FileT, output_directory: FileT) -> None:
    """Iterate through the video and dinf balls,

    TODO: Need to make better filtering.

    Args:
        video: The video to loop through looking for pool balls.
    """
    balls = []

    capture = cv2.VideoCapture(video)
    while 1:
        _, image = capture.read()

        table = OpenCVTable(image)
        table.detect_balls(True)

        # If some balls were detected save the image and record the balls
        # locations.
        if table.balls_up():
            balls.append(table.balls_up())

            output_file = str(Path(output_directory, f"frame_{len(balls)}").with_suffix(".png"))
            output_file_labeled = str(Path(output_directory, f"frame_{len(balls)}_traced").with_suffix(".png"))
            shutil.move("images/gen/detected_objects_filtered.png", output_file_labeled)
            cv2.imwrite(output_file, image)

    # Close the window
    capture.release()

    # De-allocate any associated memory usage
    cv2.destroyAllWindows()
