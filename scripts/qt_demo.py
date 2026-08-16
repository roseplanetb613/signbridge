"""Qt 测试窗口：摄像头 + 手部关键点实时跟踪验证。

用法: python scripts/qt_demo.py [--camera-id 0] [--window 60]
按 Esc 或关闭窗口退出。
"""

import argparse
import sys
import time

import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QKeyEvent, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QVBoxLayout,
    QWidget,
)

from signbridge import (
    CameraSource,
    DistanceFeatureVerifier,
    HandDetector,
    HandSequenceBuffer,
    HandShapeFeature,
    OneEuroSmoother,
)
from signbridge.core.errors import SignBridgeError
from signbridge.hands.draw import DEPTH_COLORS, draw_landmarks_depth


def _seq_last_pts(seq):
    """HandSequence 最后有效帧的 (21,3) 点（已腕点归一化，可直接提特征）。"""
    idx = np.flatnonzero(seq.valid_mask)
    if len(idx) == 0:
        return None
    return seq.data[idx[-1]]


class TrackingWindow(QMainWindow):
    """摄像头实时手部跟踪验证窗口（含特征提取可视化）。"""

    def __init__(self, camera_id: int, window_size: int) -> None:
        super().__init__()
        self.setWindowTitle("SignBridge 手部跟踪验证（特征提取）")
        try:
            self._source = CameraSource(camera_id)
            self._detector = HandDetector(max_num_hands=2)
        except SignBridgeError as exc:
            QMessageBox.critical(self, "初始化失败", str(exc))
            raise
        self._buffer = HandSequenceBuffer(
            window_size=window_size, smoother=OneEuroSmoother()
        )
        self._feature = HandShapeFeature()
        self._verifier = DistanceFeatureVerifier()
        self._last_feature: dict[int, np.ndarray] = {}   # hand_id → 上一帧特征

        self._canvas = QLabel("正在打开摄像头…")
        self._canvas.setAlignment(Qt.AlignCenter)
        self._info = QLabel("")
        self._info.setMinimumWidth(340)
        self._info.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self._feature_map = QLabel("")
        self._feature_map.setFixedSize(210, 60)
        self._feature_map.setAlignment(Qt.AlignCenter)

        central = QWidget()
        layout = QHBoxLayout(central)
        layout.addWidget(self._canvas, 1)
        side = QVBoxLayout()
        side.addWidget(self._info)
        side.addWidget(QLabel("特征热力图（210 维距离矩阵，每只手一行）"))
        side.addWidget(self._feature_map)
        layout.addLayout(side, 0)
        self.setCentralWidget(central)

        self._timer = QTimer(self)
        self._timer.setInterval(50)
        self._timer.timeout.connect(self._on_tick)
        self._timer.start()

        self._fps_start = time.monotonic()
        self._fps_frames = 0
        self._fps = 0.0
        self._total_frames = 0

    def _on_tick(self) -> None:
        try:
            frame, _, _ = next(iter(self._source))
        except StopIteration:
            self.close()
            return
        hand_frame = self._detector.detect(frame)
        seqs = self._buffer.update(hand_frame)
        self._total_frames += 1

        canvas = draw_landmarks_depth(frame, hand_frame)
        self._draw_id_labels(canvas, hand_frame, seqs)

        self._update_fps()
        self._show_frame(canvas)
        confs = self._update_features(seqs)
        self._show_info(hand_frame, seqs, confs)

    def _update_features(self, seqs) -> dict[int, float]:
        """每只手：当前帧特征 vs 上一帧特征 → 特征置信度（手形稳定度）。"""
        confs: dict[int, float] = {}
        for s in seqs:
            pts = _seq_last_pts(s)
            if pts is None:
                continue
            feat = self._feature.extract(pts)
            last = self._last_feature.get(s.hand_id)
            if last is not None:
                confs[s.hand_id] = self._verifier.verify(last, feat)
            self._last_feature[s.hand_id] = feat
        # 热力图：每只手一行（210 → 14×15 网格），JET 伪彩
        if seqs:
            vecs = [self._last_feature.get(s.hand_id) for s in seqs]
            vecs = [v for v in vecs if v is not None]
            if vecs:
                heat = np.stack(vecs)
                heat = (heat - heat.min()) / max(heat.max() - heat.min(), 1e-9)
                heat = (heat * 255).astype(np.uint8)
                heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
                heat = cv2.resize(heat, (210, 20 * len(vecs)),
                                  interpolation=cv2.INTER_NEAREST)
                rgb = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb.shape
                img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
                self._feature_map.setPixmap(QPixmap.fromImage(img.copy()))
        return confs

    def _draw_id_labels(self, canvas, hand_frame, seqs) -> None:
        for s in seqs:
            hand = next(
                (h for h in hand_frame.hands if h.handedness == s.handedness), None
            )
            if hand is None or not hand.landmarks:
                continue
            w0 = hand.landmarks[0]
            pt = (int(w0.x * canvas.shape[1]), int(w0.y * canvas.shape[0]))
            color = DEPTH_COLORS.get(s.handedness, (200, 200, 200))
            cv2.putText(
                canvas, f"id{s.hand_id} {s.handedness}",
                (pt[0] - 30, max(pt[1] - 20, 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA,
            )

    def _show_frame(self, bgr) -> None:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        self._canvas.setPixmap(QPixmap.fromImage(img.copy()))

    def _update_fps(self) -> None:
        self._fps_frames += 1
        now = time.monotonic()
        if now - self._fps_start >= 1.0:
            self._fps = self._fps_frames / (now - self._fps_start)
            self._fps_frames = 0
            self._fps_start = now

    def _show_info(self, hand_frame, seqs, confs) -> None:
        detected = "、".join(
            f"{h.handedness}({h.score:.2f})" for h in hand_frame.hands
        ) or "无"
        lines = [
            f"<b>帧:</b> {self._total_frames}　<b>FPS:</b> {self._fps:.1f}",
            f"<b>检测:</b> {detected}",
            f"<b>跟踪轨迹:</b> {len(seqs)} 条",
            "<hr>",
        ]
        for s in seqs:
            valid = int(s.valid_mask.sum())
            conf = confs.get(s.hand_id)
            conf_txt = f"特征稳定度 {conf:.2f}" if conf is not None else "特征待初始化"
            lines.append(
                f"<b>id{s.hand_id}</b> {s.handedness} · {conf_txt}<br>"
                f"序列 {len(s.data)} 帧（有效 {valid}）"
            )
        self._info.setText("<br>".join(lines))

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key_Escape:
            self.close()

    def closeEvent(self, event) -> None:
        self._timer.stop()
        try:
            self._source.close()
        finally:
            self._detector.close()
        event.accept()


def main() -> int:
    parser = argparse.ArgumentParser(description="SignBridge Qt 手部跟踪验证")
    parser.add_argument("--camera-id", type=int, default=0)
    parser.add_argument("--window", type=int, default=60)
    args = parser.parse_args()
    app = QApplication(sys.argv)
    try:
        win = TrackingWindow(args.camera_id, args.window)
    except SignBridgeError:
        return 1
    win.resize(1280, 720)
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
