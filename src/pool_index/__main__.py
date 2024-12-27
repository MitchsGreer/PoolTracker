"""Detect pool balls on an image."""
import argparse

from pool_index.table import OpenCVTable
from pool_index.dataset import download_video, iterate_through_video
from pool_index.util import DebugT

def _parser() -> argparse.ArgumentParser:
    """Argument parser for this project."""
    parser = argparse.ArgumentParser()

    # parser.add_argument("image")
    # parser.add_argument("method", choices=["opencv", "object_detection"])
    # parser.add_argument("--debug", "-d", action="store_true", default=False)

    return parser


def main() -> None:
    """This is the main entry point for this module."""
    args = _parser().parse_args()

    # if args.method == "opencv":
    #     table = OpenCVTable(args.image)
    #     table.detect_balls(DebugT.LEVEL_2, "images/gen")
    #     print("Balls up:", table.balls_up())

    # output_file = download_video("https://www.youtube.com/watch?v=vNkngrO7B9g", "video")
    # print(output_file)
    iterate_through_video("video/Shane VAN BOENING vs Jeremy SEAMAN ▸ 2023 Las Vegas Open.mp4", "images/dataset")


if __name__ == "__main__":
    main()
