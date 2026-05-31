import argparse
import cv2

from src.video.video_loader import VideoLoader
from src.calibration.roll_calibrator import RollCalibrator
from src.calibration.camera_profile import CameraProfile
from src.calibration.coordinate_transformer import CoordinateTransformer


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


def add_calibrate_parser(subparsers):
    parser = subparsers.add_parser(
        "calibrate",
        help="Create camera roll calibration profile.",
    )

    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Input video path.",
    )

    parser.add_argument(
        "--profile",
        type=str,
        required=True,
        help="Output camera profile JSON path.",
    )

    parser.add_argument(
        "--frame_idx",
        type=int,
        default=0,
        help="Frame index used for calibration.",
    )

    parser.add_argument(
        "--display_scale",
        type=float,
        default=0.75,
        help="Display scale for calibration window.",
    )

    return parser


def add_calibration_preview_parser(subparsers):
    parser = subparsers.add_parser(
        "calibration-preview",
        help="Preview camera roll correction result.",
    )

    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Input video path.",
    )

    parser.add_argument(
        "--profile",
        type=str,
        required=True,
        help="Camera profile JSON path.",
    )

    parser.add_argument(
        "--frame_idx",
        type=int,
        default=0,
        help="Frame index to preview.",
    )

    parser.add_argument(
        "--display_scale",
        type=float,
        default=0.75,
        help="Display scale for preview windows.",
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
    add_calibrate_parser(subparsers)
    add_calibration_preview_parser(subparsers)

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


def run_calibrate(args):
    calibrator = RollCalibrator(
        video_path=args.video,
        frame_idx=args.frame_idx,
        display_scale=args.display_scale,
    )

    profile = calibrator.calibrate()
    profile.save(args.profile)

    print(f"Camera profile saved to: {args.profile}")


def run_calibration_preview(args):
    profile = CameraProfile.load(args.profile)
    transformer = CoordinateTransformer(profile)

    with VideoLoader(args.video) as loader:
        result = loader.read_frame(args.frame_idx)

        if result is None:
            raise RuntimeError(f"Failed to read frame: {args.frame_idx}")

        frame_idx, frame = result

    corrected = transformer.warp_frame(frame)

    original_preview = cv2.resize(
        frame,
        None,
        fx=args.display_scale,
        fy=args.display_scale,
        interpolation=cv2.INTER_AREA,
    )

    corrected_preview = cv2.resize(
        corrected,
        None,
        fx=args.display_scale,
        fy=args.display_scale,
        interpolation=cv2.INTER_AREA,
    )

    cv2.putText(
        original_preview,
        f"Original frame {frame_idx}",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 255),
        2,
    )

    cv2.putText(
        corrected_preview,
        f"Corrected frame {frame_idx} | roll={profile.roll_angle_deg:.2f}",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 255),
        2,
    )

    cv2.imshow("Original", original_preview)
    cv2.imshow("Roll Corrected", corrected_preview)

    print()
    print("=== Calibration Preview ===")
    print(f"profile: {args.profile}")
    print(f"roll_angle_deg: {profile.roll_angle_deg:.4f}")
    print("Press any key to close preview windows.")

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    args = parse_args()

    if args.command == "video-test":
        run_video_test(args)

    elif args.command == "calibrate":
        run_calibrate(args)

    elif args.command == "calibration-preview":
        run_calibration_preview(args)

    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()