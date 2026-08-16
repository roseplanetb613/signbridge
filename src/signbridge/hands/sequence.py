"""时序序列缓冲：帧间追踪 + ID 生命周期 + 滑动窗口 + 腕点归一化。

消费第一步的 HandFrame，输出按手 ID 稳定分离的 HandSequence
（ST-GCN 输入：data 为 (T, 21, 3) 腕点归一化坐标）。
匹配与平滑均为可插拔协议（core.matching / core.smoothing）。
"""

import copy
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from signbridge.core.features import FeatureExtractor, HandShapeFeature
from signbridge.core.landmarks import HandFrame
from signbridge.core.matching import (
    FeatureHungarianMatcher,
    HandDescriptor,
    Matcher,
)
from signbridge.core.smoothing import LandmarkSmoother


@dataclass(frozen=True)
class HandSequence:
    """一只手的时序序列（滑动窗口快照）。

    data: (T, 21, 3) float32，腕点(WRIST)归一化；丢失帧为 NaN 行。
    valid_mask: (T,) bool；timestamps/frame_indices 与 data 逐行对应。
    """

    hand_id: int
    handedness: str
    data: np.ndarray = field(repr=False)
    valid_mask: np.ndarray = field(repr=False)
    timestamps: np.ndarray = field(repr=False)
    frame_indices: np.ndarray = field(repr=False)


class _Track:
    __slots__ = ("hand_id", "handedness", "centroid", "lost_count",
                 "smoother", "last_feature", "flip_count", "slots",
                 "timestamps", "frame_indices")

    def __init__(self, hand_id, handedness, centroid, smoother):
        self.hand_id = hand_id
        self.handedness = handedness
        self.centroid = centroid
        self.lost_count = 0
        self.smoother = smoother
        self.last_feature = None
        self.flip_count = 0
        self.slots: deque = deque()
        self.timestamps: deque = deque()
        self.frame_indices: deque = deque()


class HandSequenceBuffer:
    """手部时序缓冲：每帧调用 update(hand_frame)，返回当前活动手的序列。

    参数：
        window_size: 滑动窗口帧数（槽位按帧推进，含丢失占位）
        max_hands: 手数上限（当前仅用于校验提示）
        max_lost_frames: ID 失联保留帧数，超过则回收
        matcher: 可插拔帧间匹配器（默认 FeatureHungarianMatcher：
                 位置匈牙利 + 特征恢复；传 HungarianMatcher 退回纯位置）
        coordinate: "world"（米制 world_landmarks，默认）| "image"（归一化坐标）
        smoother: 可插拔平滑器实例（内部按手 deepcopy）；None 不平滑
        feature_extractor: 可插拔特征提取器（默认 HandShapeFeature；
                           供匹配器做跨位置恢复判定；None 关闭特征）
        handedness_debounce: 左右手标签防抖帧数（默认 5）——判定相反需
                            连续 ≥ 该帧数才切换标签，抑制 MediaPipe 对手形
                            的偶发左右误判；0 关闭防抖逐帧跟随
    """

    def __init__(
        self,
        window_size: int = 60,
        max_hands: int = 2,
        max_lost_frames: int = 10,
        matcher: Matcher | None = None,
        coordinate: str = "world",
        smoother: LandmarkSmoother | None = None,
        feature_extractor: FeatureExtractor | None = None,
        handedness_debounce: int = 5,
    ) -> None:
        if window_size <= 0:
            raise ValueError("window_size 必须 > 0")
        if max_lost_frames < 0:
            raise ValueError("max_lost_frames 必须 >= 0")
        if handedness_debounce < 0:
            raise ValueError("handedness_debounce 必须 >= 0")
        if coordinate not in ("world", "image"):
            raise ValueError("coordinate 必须是 'world' 或 'image'")
        self.window_size = window_size
        self.max_hands = max_hands
        self.max_lost_frames = max_lost_frames
        self.coordinate = coordinate
        self.handedness_debounce = handedness_debounce
        self._matcher = (
            matcher if matcher is not None else FeatureHungarianMatcher()
        )
        self._feature_extractor = (
            feature_extractor if feature_extractor is not None
            else HandShapeFeature()
        )
        self._smoother_factory = (
            (lambda: copy.deepcopy(smoother)) if smoother is not None else None
        )
        self._tracks: dict[int, _Track] = {}
        self._next_id = 0
        self._frame_index = 0

    @property
    def left_hand_id(self) -> int:
        """当前左手 ID；无左手时为 -1。"""
        return self._hand_id_for("Left")

    @property
    def right_hand_id(self) -> int:
        """当前右手 ID；无右手时为 -1。"""
        return self._hand_id_for("Right")

    def _hand_id_for(self, handedness: str) -> int:
        for t in self._tracks.values():
            if t.handedness == handedness:
                return t.hand_id
        return -1

    def update(self, hand_frame: HandFrame) -> tuple[HandSequence, ...]:
        """喂入一帧 HandFrame，返回当前所有活动手的 HandSequence（按 hand_id 升序）。"""
        cur = self._extract(hand_frame)
        cur_descriptors = [
            HandDescriptor(centroid=c, feature=feat) for _, c, _, feat in cur
        ]
        prev_descriptors = [
            HandDescriptor(centroid=t.centroid, feature=t.last_feature)
            for t in self._tracks.values()
        ]
        matching = self._matcher.match(cur_descriptors, prev_descriptors)
        track_list = list(self._tracks.values())
        ts = hand_frame.timestamp_ms

        for ci, pi in matching.matched:                      # 匹配对 → 续用 ID
            track = track_list[pi]
            handedness, centroid, pts, feature = cur[ci]
            track.lost_count = 0
            # handedness 防抖：判定相反需连续 ≥ debounce 帧才切换标签
            if handedness != track.handedness:
                track.flip_count += 1
                if track.flip_count >= self.handedness_debounce:
                    track.handedness = handedness
                    track.flip_count = 0
            else:
                track.flip_count = 0
            track.centroid = centroid
            track.last_feature = feature
            self._append_valid(track, pts, ts)

        for ci in matching.unmatched_current:                # 新手 → 新 ID
            handedness, centroid, pts, feature = cur[ci]
            track = _Track(
                self._next_id, handedness, centroid,
                self._smoother_factory() if self._smoother_factory else None,
            )
            track.last_feature = feature
            self._next_id += 1
            self._tracks[track.hand_id] = track
            self._append_valid(track, pts, ts)

        for pi in matching.unmatched_previous:               # 失联 → lost 计数 + 占位
            track = track_list[pi]
            track.lost_count += 1
            self._append_invalid(track, ts)

        for hand_id in [
            t.hand_id for t in self._tracks.values()
            if t.lost_count > self.max_lost_frames
        ]:
            del self._tracks[hand_id]

        self._frame_index += 1
        return tuple(
            self._to_sequence(t)
            for t in sorted(self._tracks.values(), key=lambda t: t.hand_id)
        )

    def reset(self) -> None:
        """清空所有轨迹与窗口，帧计数归零。"""
        self._tracks.clear()
        self._next_id = 0
        self._frame_index = 0

    # ---- 内部 ----

    def _extract(self, hand_frame: HandFrame):
        """提取当前帧每只手：[(handedness, 质心, pts(21,3), feature|None)]。"""
        out = []
        for hand in hand_frame.hands:
            lms = hand.world_landmarks if self.coordinate == "world" else hand.landmarks
            if len(lms) < 21:
                continue
            pts = np.array([[lm.x, lm.y, lm.z] for lm in lms], dtype=np.float32)
            feature = (
                self._feature_extractor.extract(pts)
                if self._feature_extractor is not None else None
            )
            out.append((hand.handedness, pts[:, :2].mean(axis=0), pts, feature))
        return out

    def _append_valid(self, track: _Track, pts: np.ndarray, ts: int) -> None:
        pts = pts - pts[0]  # 腕点归一化（WRIST 为原点）
        if track.smoother is not None:
            pts = track.smoother.update(pts)
        track.slots.append(pts.copy())
        track.timestamps.append(ts)
        track.frame_indices.append(self._frame_index)
        self._trim(track)

    def _append_invalid(self, track: _Track, ts: int) -> None:
        track.slots.append(None)
        track.timestamps.append(ts)
        track.frame_indices.append(self._frame_index)
        if track.smoother is not None:
            track.smoother.update(None)
        self._trim(track)

    def _trim(self, track: _Track) -> None:
        while len(track.slots) > self.window_size:
            track.slots.popleft()
            track.timestamps.popleft()
            track.frame_indices.popleft()

    def _to_sequence(self, track: _Track) -> HandSequence:
        t = len(track.slots)
        data = np.full((t, 21, 3), np.nan, dtype=np.float32)
        valid = np.zeros(t, dtype=bool)
        for i, slot in enumerate(track.slots):
            if slot is not None:
                data[i] = slot
                valid[i] = True
        return HandSequence(
            hand_id=track.hand_id,
            handedness=track.handedness,
            data=data,
            valid_mask=valid,
            timestamps=np.array(track.timestamps, dtype=np.int64),
            frame_indices=np.array(track.frame_indices, dtype=np.int64),
        )
