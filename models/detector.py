import cv2
import numpy as np

class Detector:
    def __init__(self, img_width=848, img_height=480):
        self.img_width = img_width
        self.img_height = img_height
        
        self.roi_y_min = 0
        self.roi_y_max = img_height
        
        self.raw = None
        self.binary = None 
        self.ball_pos = None
        self.projection = None  

    def process_image(self, frame, roi_height, block_size, c_val):
        self.raw = frame
        
        self.roi_y_min = max(0, self.img_height // 2 - roi_height // 2)
        self.roi_y_max = min(self.img_height, self.img_height // 2 + roi_height // 2)
        
        roi = frame[self.roi_y_min:self.roi_y_max, :]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)     
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        binary = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
            cv2.THRESH_BINARY_INV, block_size, c_val
        )      
        kernel = np.ones((7, 7), np.uint8)
        self.binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        return self.binary
        
    def find_ball(self, binary, proj_min_val):
        self.projection = np.sum(binary // 255, axis=0)
        
        if np.max(self.projection) > proj_min_val:
            # 1. 先粗略找到最高峰的位置
            max_idx = int(np.argmax(self.projection))
            
            # 2. 划定一个局部窗口（比如峰值左右 15 像素）
            window_half_width = 15
            start = max(0, max_idx - window_half_width)
            end = min(len(self.projection), max_idx + window_half_width + 1)
            
            # 3. 截取局部波形和对应的索引
            local_proj = self.projection[start:end]
            local_indices = np.arange(start, end, dtype=np.float32)
            
            # 4. 只在这个干净的局部波峰内算加权平均（亚像素精度）
            if np.sum(local_proj) > 0:
                cx = np.average(local_indices, weights=local_proj)
            else:
                cx = max_idx
            
            cy = self.roi_y_min + (self.roi_y_max - self.roi_y_min) // 2
            self.ball_pos = (cx, cy)
            return self.ball_pos
            
        self.ball_pos = None
        return None

    def detect(self, frame, roi_height, block_size, c_val, proj_min_val):
        bin_img = self.process_image(frame, roi_height, block_size, c_val)
        return self.find_ball(bin_img, proj_min_val)
    
    def display(self, dis):
        if self.raw is None:
            return None, self.binary, None
            
        vis = self.raw.copy()
        proj_canvas = None
        
        if dis == 1:
            cv2.line(vis, (0, self.roi_y_min), (self.img_width, self.roi_y_min), (0, 255, 255), 2)
            cv2.line(vis, (0, self.roi_y_max), (self.img_width, self.roi_y_max), (0, 255, 255), 2)
            if self.ball_pos is not None:
                cx, cy = int(self.ball_pos[0]), int(self.ball_pos[1])
                cv2.circle(vis, (cx, cy), 10, (0, 255, 0), 2)
                cv2.circle(vis, (cx, cy), 3, (0, 0, 255), -1)
                cv2.line(vis, (cx, self.roi_y_min), (cx, self.roi_y_max), (255, 0, 0), 1)
                
            if self.projection is not None:
                proj_canvas = np.zeros((150, self.img_width, 3), dtype=np.uint8)
                max_p = np.max(self.projection) + 1e-5
                norm_proj = (self.projection / max_p * 120).astype(np.int32)
                
                for x in range(1, len(norm_proj)):
                    cv2.line(
                        proj_canvas,
                        (x - 1, 140 - norm_proj[x - 1]),
                        (x, 140 - norm_proj[x]),
                        (255, 255, 255),
                        1,
                    )
                if self.ball_pos is not None:
                    bx = int(self.ball_pos[0])
                    cv2.line(proj_canvas, (bx, 0), (bx, 150), (0, 0, 255), 2)

        return vis, self.binary, proj_canvas