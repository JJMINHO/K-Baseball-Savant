import cv2
from dataclasses import dataclass
from typing import Iterator, Optional, Tuple, Any


@dataclass
class VideoMetadata:
    """
    영상 기본 메타데이터.
    """

    video_path: str
    fps: float
    width: int
    height: int
    frame_count: int
    duration_sec: float


class VideoLoader:
    """
    OpenCV 기반 영상 로더.

    역할:
        1. 영상 파일 열기
        2. 영상 메타데이터 확인
        3. 프레임 순차 읽기
        4. 특정 프레임 읽기
    """

    def __init__(self, video_path: str):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if self.fps > 0:
            self.duration_sec = self.frame_count / self.fps
        else:
            self.duration_sec = 0.0

    def get_metadata(self) -> VideoMetadata:
        """
        영상 메타데이터 반환.
        """

        return VideoMetadata(
            video_path=self.video_path,
            fps=self.fps,
            width=self.width,
            height=self.height,
            frame_count=self.frame_count,
            duration_sec=self.duration_sec,
        )

    def read_frame(self, frame_idx: int) -> Optional[Tuple[int, Any]]:
        """
        특정 프레임 번호의 frame을 읽는다.

        Args:
            frame_idx:
                읽고 싶은 프레임 번호.

        Returns:
            성공 시:
                (frame_idx, frame)

            실패 시:
                None
        """

        if frame_idx < 0 or frame_idx >= self.frame_count:
            return None

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()

        if not ret or frame is None:
            return None

        return frame_idx, frame

    def iter_frames(
        self,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
        step: int = 1,
    ) -> Iterator[Tuple[int, Any]]:
        """
        프레임을 순차적으로 읽는 iterator.

        Args:
            start_frame:
                시작 프레임 번호.

            end_frame:
                종료 프레임 번호.
                None이면 영상 마지막 프레임까지 읽음.

            step:
                몇 프레임마다 읽을지.
                step=1이면 모든 프레임.
                step=300이면 300프레임마다 읽음.

        Yields:
            (frame_idx, frame)
        """

        if step <= 0:
            raise ValueError("step must be >= 1")

        if end_frame is None:
            end_frame = self.frame_count - 1

        start_frame = max(0, start_frame)
        end_frame = min(self.frame_count - 1, end_frame)

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        current_idx = start_frame

        while current_idx <= end_frame:
            ret, frame = self.cap.read()

            if not ret or frame is None:
                break

            if (current_idx - start_frame) % step == 0:
                yield current_idx, frame

            current_idx += 1

    def release(self):
        """
        VideoCapture 자원 해제.
        """

        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def __iter__(self):
        """
        for frame_idx, frame in loader:
            ...
        형태로 사용 가능.
        """

        return self.iter_frames()

    def __enter__(self):
        """
        with VideoLoader(video_path) as loader:
            ...
        형태로 사용 가능.
        """

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()