import json
import os
from dataclasses import asdict, dataclass
from typing import List, Tuple


@dataclass
class CameraProfile:
    """
    구장/카메라별 보정 정보를 저장하는 데이터 클래스.

    현재 1차 구현에서는 roll correction만 저장한다.
    """

    video_path: str
    frame_width: int
    frame_height: int
    roll_angle_deg: float
    reference_points: List[Tuple[float, float]]

    def save(self, output_path: str) -> None:
        """
        CameraProfile을 JSON 파일로 저장한다.
        """

        output_dir = os.path.dirname(output_path)

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        data = asdict(self)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, profile_path: str) -> "CameraProfile":
        """
        JSON 파일에서 CameraProfile을 불러온다.
        """

        if not os.path.exists(profile_path):
            raise FileNotFoundError(f"Camera profile not found: {profile_path}")

        with open(profile_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return cls(
            video_path=data["video_path"],
            frame_width=int(data["frame_width"]),
            frame_height=int(data["frame_height"]),
            roll_angle_deg=float(data["roll_angle_deg"]),
            reference_points=[
                (float(x), float(y))
                for x, y in data["reference_points"]
            ],
        )