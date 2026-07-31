import cv2
import numpy as np

class Detector:
    def __init__(self, img_width=848, img_height=480):
        self.img_width = img_width
        self.img_height = img_height
        
        self.ry1 = 0
        self.ry2 = img_height
        
        self.raw = None
        self.binary = None 
        self.ball_pos = None
        self.projection = None  

    def process_image(self, frame, ry1, ry2, block_size, c_val):
        self.raw = frame
        
        self.ry1 = min(ry1, ry2)
        self.ry2 = max(ry1, ry2)
        if self.ry2 <= self.ry1: 
            self.ry2 = self.ry1 + 1
            
        roi = frame[self.ry1:self.ry2, :]
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
            max_idx = int(np.argmax(self.projection))
            
            window_half_width = 15
            start = max(0, max_idx - window_half_width)
            end = min(len(self.projection), max_idx + window_half_width + 1)
            
            local_proj = self.projection[start:end]
            local_indices = np.arange(start, end, dtype=np.float32)
            
            if np.sum(local_proj) > 0:
                cx = np.average(local_indices, weights=local_proj)
            else:
                cx = max_idx
                
            cy = self.ry1 + (self.ry2 - self.ry1) // 2
            self.ball_pos = (cx, cy)
            return self.ball_pos
            
        self.ball_pos = None
        return None

    def detect(self, frame, ry1, ry2, block_size, c_val, proj_min_val):
        bin_img = self.process_image(frame, ry1, ry2, block_size, c_val)
        return self.find_ball(bin_img, proj_min_val)
    
    def display(self, dis):
        if self.raw is None:
            return None, self.binary, None
            
        vis = self.raw.copy()
        proj_canvas = None
        
        if dis == 1:
            cv2.rectangle(vis, (0, self.ry1), (self.img_width, self.ry2), (0, 255, 255), 2)
            
            if self.ball_pos is not None:
                cx, cy = int(self.ball_pos[0]), int(self.ball_pos[1])
                cv2.circle(vis, (cx, cy), 10, (0, 255, 0), 2)
                cv2.circle(vis, (cx, cy), 3, (0, 0, 255), -1)
                cv2.line(vis, (cx, self.ry1), (cx, self.ry2), (255, 0, 0), 1)
                
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