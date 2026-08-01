import numpy as np
from .Kalman import KalmanFilter
from enum import IntEnum

class Status(IntEnum):
    LOST = 0
    TMP_LOST = 2
    TRACK = 3

class Tracker:
    def __init__(self, img_width=848, cm_per_pixel=0.052, use_kf=True):
        self.img_width = img_width
        self.cm_per_pixel = cm_per_pixel
        self.use_kf = use_kf
        self.kf_x = KalmanFilter(q_scale=1.0, r_scale=0.2)
        self.lost_count = 0
        self.frame_lost_tol = 5
        self.status = Status.LOST

    def track(self, ball_center, dt, zero_x):
        cx = 0.0
        filtered_v = 0.0
        raw_offset_x = 0.0

        if ball_center is not None:
            # 已反转符号：从 (ball_center[0] - zero_x) 改为 (zero_x - ball_center[0])
            raw_offset_x = (zero_x - ball_center[0]) * self.cm_per_pixel

        if self.use_kf:
            if ball_center is not None:
                self.lost_count = 0
                if self.status == Status.LOST:
                    self.status = Status.TRACK
                    self.kf_x.reset()
                else:
                    self.status = Status.TRACK

                self.kf_x.predict(dt)
                cx, filtered_v = self.kf_x.update(raw_offset_x)
            else:
                self.lost_count += 1
                if self.lost_count <= self.frame_lost_tol:
                    self.status = Status.TMP_LOST
                    cx, filtered_v = self.kf_x.predict(dt)
                else:
                    self.status = Status.LOST
                    self.kf_x.reset()
        else:
            if ball_center is not None:
                cx, filtered_v = raw_offset_x, 0.0
                self.status = Status.TRACK
            else:
                cx, filtered_v = 0.0, 0.0
                self.status = Status.LOST

        return cx, filtered_v, self.status
