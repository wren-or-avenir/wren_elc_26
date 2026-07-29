import cv2
import numpy as np

class Detector:
    def __init__(self, img_width=640, img_height=480):
        self.img_width = img_width
        self.img_height = img_height
        
        self.roi_x_min = 0
        self.roi_x_max = img_width
        
        self.raw = None
        self.binary = None 
        self.ball_pos = None

    def process_image(self, frame, roi_width):
        self.raw = frame
        
        # 根据传入的 roi_width 动态计算边界
        self.roi_x_min = max(0, self.img_width // 2 - roi_width // 2)
        self.roi_x_max = min(self.img_width, self.img_width // 2 + roi_width // 2)
        
        roi = frame[:, self.roi_x_min:self.roi_x_max]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)     
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 核心修正 1: 反相二值化 (THRESH_BINARY_INV) 
        # 白色的管子变黑底，黑色的钢珠变白块
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)        
        
        # 核心修正 2: 形态学闭运算 (补洞)
        # 填补钢珠中间因为高光反光产生的破洞，使其变成实心图形
        kernel = np.ones((7, 7), np.uint8)
        self.binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        return self.binary
        
    def find_ball(self, binary):
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if hierarchy is None or len(contours) == 0:
            self.ball_pos = None
            return None

        best_ball = None
        max_area = 0
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # 这里的面积阈值可能需要根据补洞后的实际大小微调
            if 50 < area < 2000:
                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = float(w) / max(h, 1)
                if 0.5 < aspect_ratio < 2.0:
                    if area > max_area:
                        max_area = area
                        M = cv2.moments(cnt)
                        if M["m00"] != 0:
                            cx_roi = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])
                            cx = cx_roi + self.roi_x_min
                            best_ball = (cx, cy)
                            
        self.ball_pos = best_ball
        return best_ball

    def _draw_annotations(self, image):
        # 画出动态的 ROI 边界
        cv2.line(image, (self.roi_x_min, 0), (self.roi_x_min, self.img_height), (0, 255, 255), 2)
        cv2.line(image, (self.roi_x_max, 0), (self.roi_x_max, self.img_height), (0, 255, 255), 2)
        
        if self.ball_pos is not None:
            cx, cy = self.ball_pos
            cv2.circle(image, (cx, cy), 10, (0, 255, 0), 2)
            cv2.circle(image, (cx, cy), 3, (0, 0, 255), -1)

    def draw(self, image):
        if image is None:
            return image
        self._draw_annotations(image)
        return image

    # 接口改变：现在需要传入 roi_width
    def detect(self, frame, roi_width):
        bin_img = self.process_image(frame, roi_width)
        return self.find_ball(bin_img)
    
    def display(self, dis):
        if self.raw is None:
            return None, self.binary
        vis = self.raw.copy()
        res = None
        if dis == 1:
            res = self.draw(vis)
        return res, self.binary