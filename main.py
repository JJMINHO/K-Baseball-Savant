import argparse
import cv2

from src.video.video_loader import VideoLoader


def add_video_test_parser(subparsers):
    parser = subparsers.add_parser(
        "video-test",
        help="Test OpenCV video loading and frame iteration.",
    )

    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Input video path.",
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help="Show sampled frames in OpenCV window.",
    )

    parser.add_argument(
        "--step",
        type=int,
        default=300,
        help="Frame sampling step for preview.",
    )

    parser.add_argument(
        "--max_frames",
        type=int,
        default=10,
        help="Maximum number of sampled frames to print/show.",
    )

    return parser


def parse_args():
    parser = argparse.ArgumentParser(
        description="K-Baseball Savant / PitcherNet-lite"
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    add_video_test_parser(subparsers)

    return parser.parse_args()


def run_video_test(args):
    print("=== Video Loader Test ===")
    print(f"video_path: {args.video}")
    print()

    with VideoLoader(args.video) as loader:
        metadata = loader.get_metadata()

        print("=== Video Metadata ===")
        print(f"video_path: {metadata.video_path}")
        print(f"fps: {metadata.fps}")
        print(f"width: {metadata.width}")
        print(f"height: {metadata.height}")
        print(f"frame_count: {metadata.frame_count}")
        print(f"duration_sec: {metadata.duration_sec:.2f}")
        print()

        print("=== Specific Frame Read Test ===")

        test_indices = [
            0,
            metadata.frame_count // 2,
            metadata.frame_count - 1,
        ]

        for idx in test_indices:
            result = loader.read_frame(idx)

            if result is None:
                print(f"frame {idx}: failed")
            else:
                frame_idx, frame = result
                print(f"frame {frame_idx}: shape={frame.shape}")

        print()
        print("=== Frame Iteration Test ===")

        count = 0

        for frame_idx, frame in loader.iter_frames(step=args.step):
            print(f"sample {count + 1}: frame_idx={frame_idx}, shape={frame.shape}")

            if args.show:
                preview = frame.copy()

                cv2.putText(
                    preview,
                    f"Frame: {frame_idx}",
                    (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.5,
                    (0, 255, 255),
                    3,
                )

                cv2.imshow("VideoLoader Preview", preview)

                key = cv2.waitKey(0)

                if key == ord("q"):
                    break

            count += 1

            if count >= args.max_frames:
                break

    cv2.destroyAllWindows()

    print()
    print("Video loader test finished.")


def main():
    args = parse_args()

    if args.command == "video-test":
        run_video_test(args)
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()