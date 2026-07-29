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

    # 新增 block_size 和 c_val 参数
    def process_image(self, frame, roi_width, block_size, c_val):
        self.raw = frame
        
        self.roi_x_min = max(0, self.img_width // 2 - roi_width // 2)
        self.roi_x_max = min(self.img_width, self.img_width // 2 + roi_width // 2)
        
        roi = frame[:, self.roi_x_min:self.roi_x_max]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)     
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 核心替换：使用局部自适应阈值，ADAPTIVE_THRESH_MEAN_C + THRESH_BINARY_INV
        # 这将无视大面积的光照渐变，强行把比周围稍暗的钢珠给提亮成白块
        binary = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
            cv2.THRESH_BINARY_INV, block_size, c_val
        )      
        
        # 形态学闭运算：补洞
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

    # 接口改变：同样传入新参数
    def detect(self, frame, roi_width, block_size, c_val):
        bin_img = self.process_image(frame, roi_width, block_size, c_val)
        return self.find_ball(bin_img)
    
    def display(self, dis):
        if self.raw is None:
            return None, self.binary
        vis = self.raw.copy()
        res = None
        if dis == 1:
            res = self.draw(vis)
        return res, self.binary