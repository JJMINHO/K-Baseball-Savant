import math
from typing import Tuple

import cv2
import numpy as np

from src.calibration.camera_profile import CameraProfile


class CoordinateTransformer:
    """
    CameraProfile을 이용해 좌표 또는 프레임을 보정하는 클래스.

    현재는 camera roll correction만 지원한다.
    """

    def __init__(self, profile: CameraProfile):
        self.profile = profile

        self.center = (
            profile.frame_width / 2.0,
            profile.frame_height / 2.0,
        )

        # 기준선의 기울기를 제거해야 하므로 -roll_angle 만큼 회전한다.
        self.correction_angle_deg = -profile.roll_angle_deg

        self.rotation_matrix = cv2.getRotationMatrix2D(
            center=self.center,
            angle=self.correction_angle_deg,
            scale=1.0,
        )

    def transform_point(
        self,
        x: float,
        y: float,
    ) -> Tuple[float, float]:
        """
        단일 점 좌표에 roll correction을 적용한다.
        """

        point = np.array([x, y, 1.0], dtype=np.float32)
        transformed = self.rotation_matrix @ point

        return float(transformed[0]), float(transformed[1])

    def transform_points(self, points):
        """
        여러 점 좌표에 roll correction을 적용한다.
        """

        return [
            self.transform_point(x, y)
            for x, y in points
        ]

    def warp_frame(self, frame):
        """
        frame 전체에 roll correction을 적용한다.
        주로 calibration preview 또는 debug 영상 확인용.
        """

        return cv2.warpAffine(
            frame,
            self.rotation_matrix,
            (self.profile.frame_width, self.profile.frame_height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )

    def inverse_transform_point(
        self,
        x: float,
        y: float,
    ) -> Tuple[float, float]:
        """
        보정 좌표를 원본 좌표로 되돌린다.
        """

        inverse_matrix = cv2.invertAffineTransform(self.rotation_matrix)

        point = np.array([x, y, 1.0], dtype=np.float32)
        transformed = inverse_matrix @ point

        return float(transformed[0]), float(transformed[1])