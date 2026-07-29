import math
import numpy as np
from .Kalman import KalmanFilter
import time
from enum import IntEnum
import cv2

class Status(IntEnum):
    LOST = 0
    TMP_LOST = 2
    TRACK = 3

class Tracker:
    def __init__(self, img_width=640, img_height=480, cm_per_pixel=0.052, use_kf=True):
        self.img_width = img_width
        self.img_height = img_height
        self.cm_per_pixel = cm_per_pixel
        self.use_kf = use_kf
        
        # 像以前一样，调用独立的 KalmanFilter
        # Q:0.1 R:1.0 是针对钢珠高速运动防噪的推荐参数
        self.kf_y = KalmanFilter(q_scale=0.1, r_scale=1.0)
        
        self.lost_count = 0
        self.frame_lost_tol = 5
        self.last_time = None
        self.status = Status.LOST

        self.raw = None

    def time_diff(self):
        current_time = time.time_ns()
        if self.last_time is None:
            self.last_time = current_time
            return 0.033
        else:
            diff = (current_time - self.last_time)
            self.last_time = current_time
            return diff / 1e9

    def filter(self, target_center):
        cy = 0.0
        filtered_v = 0.0
        dt = self.time_diff()

        # 计算原始的 Y 轴偏差 (cm)
        raw_offset_y = 0.0
        if target_center is not None:
            raw_offset_y = (target_center[1] - self.img_height / 2.0) * self.cm_per_pixel

        if self.use_kf:
            if target_center is not None:
                self.lost_count = 0
                if self.status == Status.LOST:
                    self.status = Status.TRACK
                    self.kf_y.reset()
                else:
                    self.status = Status.TRACK
                
                # 预测与更新 (完全利用你原有的 Kalman.py 逻辑)
                self.kf_y.predict(dt)
                update_y, update_vy = self.kf_y.update(raw_offset_y)
                cy, filtered_v = update_y, update_vy
            else:
                self.lost_count += 1
                if self.lost_count <= self.frame_lost_tol:
                    self.status = Status.TMP_LOST
                    pred_y, pred_vy = self.kf_y.predict(dt)
                    cy, filtered_v = pred_y, pred_vy
                else:
                    self.status = Status.LOST
                    self.kf_y.reset()
                    cy, filtered_v = 0.0, 0.0
        else:
            if target_center is not None:
                cy = raw_offset_y
                filtered_v = 0.0
                self.status = Status.TRACK
            else:
                cy, filtered_v = 0.0, 0.0
                self.status = Status.LOST

        return cy, filtered_v

    def track(self, ball_center):
        """对外接口"""
        filtered_y, filtered_v = self.filter(ball_center)
        return filtered_y, filtered_v, self.status

    def display(self, dis=1, ball_pos=None):
        if self.raw is None:
            return None
        vis = self.raw.copy()
        if dis == 1:
            cx_img, cy_img = self.img_width // 2, self.img_height // 2
            
            # 画出画面中心点和水平线
            cv2.circle(vis, (cx_img, cy_img), 5, (0, 165, 255), -1)
            cv2.line(vis, (0, cy_img), (self.img_width, cy_img), (0, 165, 255), 1)

            if ball_pos is not None:
                cv2.circle(vis, ball_pos, 8, (0, 255, 0), -1)
                cv2.line(vis, (cx_img, cy_img), (cx_img, ball_pos[1]), (255, 0, 0), 2)
        return vis