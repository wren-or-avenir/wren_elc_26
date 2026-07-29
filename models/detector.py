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

    def process_image(self, frame, roi_width, block_size, c_val):
        self.raw = frame
        
        self.roi_x_min = max(0, self.img_width // 2 - roi_width // 2)
        self.roi_x_max = min(self.img_width, self.img_width // 2 + roi_width // 2)
        
        roi = frame[:, self.roi_x_min:self.roi_x_max]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)     
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 局部自适应阈值抗阴影
        binary = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
            cv2.THRESH_BINARY_INV, block_size, c_val
        )      
        
        # 闭运算补满高光破洞
        kernel = np.ones((7, 7), np.uint8)
        self.binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        return self.binary
        
    def find_ball(self, binary, proj_min_val):
        # 核心：Y轴一维投影打击，直接将每行像素/255后求和，得出该行白点个数
        y_projection = np.sum(binary // 255, axis=1)
        max_val = np.max(y_projection)
        
        if max_val > proj_min_val:
            cy = int(np.argmax(y_projection))
            # 我们只关心Y轴运动，X轴固定在ROI中心即可
            cx = self.roi_x_min + (self.roi_x_max - self.roi_x_min) // 2
            self.ball_pos = (cx, cy)
            return self.ball_pos
            
        self.ball_pos = None
        return None

    def _draw_annotations(self, image):
        cv2.line(image, (self.roi_x_min, 0), (self.roi_x_min, self.img_height), (0, 255, 255), 2)
        cv2.line(image, (self.roi_x_max, 0), (self.roi_x_max, self.img_height), (0, 255, 255), 2)
        
        if self.ball_pos is not None:
            cx, cy = self.ball_pos
            cv2.circle(image, (cx, cy), 10, (0, 255, 0), 2)
            cv2.circle(image, (cx, cy), 3, (0, 0, 255), -1)
            # 画一根横贯ROI的蓝线，标明当前投影极值所在的行
            cv2.line(image, (self.roi_x_min, cy), (self.roi_x_max, cy), (255, 0, 0), 1)

    def draw(self, image):
        if image is None:
            return image
        self._draw_annotations(image)
        return image

    def detect(self, frame, roi_width, block_size, c_val, proj_min_val):
        bin_img = self.process_image(frame, roi_width, block_size, c_val)
        return self.find_ball(bin_img, proj_min_val)
    
    def display(self, dis):
        if self.raw is None:
            return None, self.binary
        vis = self.raw.copy()
        res = None
        if dis == 1:
            res = self.draw(vis)
        return res, self.binary