import math
from typing import List, Tuple

import cv2

from src.calibration.camera_profile import CameraProfile
from src.video.video_loader import VideoLoader


class RollCalibrator:
    """
    영상에서 기준선 2점을 클릭하여 camera roll angle을 계산한다.

    사용법:
        calibrator = RollCalibrator(video_path)
        profile = calibrator.calibrate()
        profile.save(profile_path)
    """

    def __init__(
        self,
        video_path: str,
        frame_idx: int = 0,
        window_name: str = "Roll Calibration",
        display_scale: float = 0.75,
    ):
        self.video_path = video_path
        self.frame_idx = frame_idx
        self.window_name = window_name
        self.display_scale = display_scale

        self.clicked_points: List[Tuple[float, float]] = []
        self.original_frame = None
        self.display_frame = None

    def calibrate(self) -> CameraProfile:
        """
        첫 프레임 또는 지정 프레임을 표시하고,
        사용자가 기준선 2점을 클릭하면 roll angle을 계산한다.
        """

        with VideoLoader(self.video_path) as loader:
            metadata = loader.get_metadata()
            result = loader.read_frame(self.frame_idx)

            if result is None:
                raise RuntimeError(f"Failed to read frame: {self.frame_idx}")

            _, frame = result

        self.original_frame = frame
        self.display_frame = self._resize_for_display(frame)

        self.clicked_points = []

        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)

        print()
        print("=== Camera Roll Calibration ===")
        print("기준선으로 사용할 두 점을 클릭하세요.")
        print("예: 홈플레이트 기준선, 마운드-홈 방향 기준선, 화면상 수평이어야 하는 선")
        print()
        print("조작법:")
        print("- 마우스 왼쪽 클릭: 점 선택")
        print("- r: 선택 초기화")
        print("- q 또는 ESC: 취소")
        print()

        while True:
            canvas = self.display_frame.copy()
            self._draw_instruction(canvas)
            self._draw_clicked_points(canvas)

            cv2.imshow(self.window_name, canvas)

            key = cv2.waitKey(20) & 0xFF

            if len(self.clicked_points) >= 2:
                break

            if key == ord("r"):
                self.clicked_points = []
                print("Selected points reset.")

            if key == ord("q") or key == 27:
                cv2.destroyWindow(self.window_name)
                raise RuntimeError("Calibration cancelled by user.")

        cv2.destroyWindow(self.window_name)

        p1, p2 = self.clicked_points[:2]
        roll_angle_deg = self._compute_roll_angle_deg(p1, p2)

        print()
        print("=== Calibration Result ===")
        print(f"point_1: {p1}")
        print(f"point_2: {p2}")
        print(f"roll_angle_deg: {roll_angle_deg:.4f}")
        print()

        return CameraProfile(
            video_path=self.video_path,
            frame_width=metadata.width,
            frame_height=metadata.height,
            roll_angle_deg=roll_angle_deg,
            reference_points=[p1, p2],
        )

    def _resize_for_display(self, frame):
        """
        화면에 표시하기 위해 frame을 축소한다.
        클릭 좌표는 원본 좌표계로 다시 환산한다.
        """

        if self.display_scale == 1.0:
            return frame.copy()

        return cv2.resize(
            frame,
            None,
            fx=self.display_scale,
            fy=self.display_scale,
            interpolation=cv2.INTER_AREA,
        )

    def _mouse_callback(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        if len(self.clicked_points) >= 2:
            return

        original_x = x / self.display_scale
        original_y = y / self.display_scale

        point = (float(original_x), float(original_y))
        self.clicked_points.append(point)

        print(f"Clicked point {len(self.clicked_points)}: {point}")

    def _draw_instruction(self, canvas):
        lines = [
            "Click 2 points for camera roll calibration",
            "r: reset | q/ESC: cancel",
        ]

        y = 35
        for line in lines:
            cv2.putText(
                canvas,
                line,
                (30, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )
            y += 35

    def _draw_clicked_points(self, canvas):
        for idx, point in enumerate(self.clicked_points):
            x = int(point[0] * self.display_scale)
            y = int(point[1] * self.display_scale)

            cv2.circle(canvas, (x, y), 8, (0, 0, 255), -1)

            cv2.putText(
                canvas,
                f"P{idx + 1}",
                (x + 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )

        if len(self.clicked_points) == 2:
            p1 = self.clicked_points[0]
            p2 = self.clicked_points[1]

            x1 = int(p1[0] * self.display_scale)
            y1 = int(p1[1] * self.display_scale)
            x2 = int(p2[0] * self.display_scale)
            y2 = int(p2[1] * self.display_scale)

            cv2.line(canvas, (x1, y1), (x2, y2), (0, 0, 255), 3)

    @staticmethod
    def _compute_roll_angle_deg(
        p1: Tuple[float, float],
        p2: Tuple[float, float],
    ) -> float:
        """
        기준선의 기울기를 계산한다.

        이미지 좌표계에서:
            x는 오른쪽으로 증가
            y는 아래로 증가

        두 점을 이은 선이 수평에서 얼마나 기울었는지를 계산한다.

        반환값:
            roll_angle_deg

        이후 좌표 보정 단계에서는 이 각도의 반대 방향으로 회전시켜
        기준선이 수평이 되도록 만든다.
        """

        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]

        angle_rad = math.atan2(dy, dx)
        angle_deg = math.degrees(angle_rad)

        return float(angle_deg)