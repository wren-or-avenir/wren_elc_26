import cv2
import time
import numpy as np
from models.cam import Camera
from models.detector import Detector
from models.tracker import Tracker, Status
from models.uart import UartDev

camera_index = 4             
uart_port = '/dev/ttyACM0'      
use_kf = True           
show_windows = 1     

camera = Camera(index=camera_index, width=640, height=480)
detector = Detector(img_width=640, img_height=480)
tracker = Tracker(img_height=480, use_kf=use_kf) 
uart = UartDev(port=uart_port, baudrate=115200)

brake_th = 1.0 

def nothing(x): pass

def init_board():
    cv2.namedWindow('Controls', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Controls', 300, 250) # 调大一点，装得下新滑块
    cv2.namedWindow('DETECTOR', cv2.WINDOW_FREERATIO)  
    cv2.namedWindow('BIN', cv2.WINDOW_FREERATIO)      
    cv2.namedWindow('Tracker', cv2.WINDOW_FREERATIO)  

    cv2.createTrackbar('brake_th', 'Controls', 10, 100, nothing) 
    cv2.createTrackbar('cm_px', 'Controls', 52, 200, nothing) 
    cv2.createTrackbar('roi_w', 'Controls', 160, 640, nothing) 
    
    # 新增：自适应阈值参数滑块
    cv2.createTrackbar('blk_size', 'Controls', 51, 201, nothing) 
    cv2.createTrackbar('C_val', 'Controls', 15, 100, nothing) 
    
    cv2.createTrackbar('show', 'Controls', 1, 1, nothing)

def update_params():
    global brake_th, show_windows
    brake_th = cv2.getTrackbarPos('brake_th', 'Controls') / 10.0
    tracker.cm_per_pixel = cv2.getTrackbarPos('cm_px', 'Controls') / 1000.0
    show_windows = cv2.getTrackbarPos('show', 'Controls')
    
    roi_width = cv2.getTrackbarPos('roi_w', 'Controls')
    if roi_width < 10: 
        roi_width = 10
        
    # 获取自适应阈值参数
    block_size = cv2.getTrackbarPos('blk_size', 'Controls')
    c_val = cv2.getTrackbarPos('C_val', 'Controls')
    
    # OpenCV 的 adaptiveThreshold 规定 block_size 必须为奇数且 >= 3
    if block_size < 3: 
        block_size = 3
    if block_size % 2 == 0: 
        block_size += 1
        
    return roi_width, block_size, c_val

def main():
    global show_windows, brake_th
    print("视觉平衡球系统启动... 按 'q' 键退出。")
    init_board()
    prev_time = time.time()

    try:
        while True:
            ret, frame = camera.read()
            if not ret: continue

            # 接收返回的自适应阈值参数
            roi_width, block_size, c_val = update_params()

            # 一并传入 detect
            ball_pos = detector.detect(frame, roi_width, block_size, c_val)
            
            y_offset, y_vel, status = tracker.track(ball_pos)
            
            curr_time = time.time()
            loop_dt = max(curr_time - prev_time, 1e-6)
            fps = 1.0 / loop_dt
            prev_time = curr_time

            send_status = 0
            if status in [Status.TRACK, Status.TMP_LOST]:
                if abs(y_offset) < brake_th:
                    send_status = 1
                else:
                    send_status = 2
            
            uart.send_data(y_offset, y_vel, send_status)
            
            info = ""
            if status == Status.TRACK:
                info = f"[TRACK] dy:{y_offset:>6.2f}cm vy:{y_vel:>6.2f}cm/s 状态:{send_status}"
            elif status == Status.TMP_LOST:
                info = f"[PRED]  dy:{y_offset:>6.2f}cm vy:{y_vel:>6.2f}cm/s 状态:{send_status}"
            else:
                info = "[LOST] searching..."

            print(f"FPS: {fps:.1f} | {info}")

            detector.raw = frame
            tracker.raw = frame

            if show_windows == 1:
                vis_det, bin_img = detector.display(dis=1)
                vis_trk = tracker.display(dis=1, ball_pos=ball_pos)
                
                if vis_det is not None: cv2.imshow("DETECTOR", vis_det)
                if bin_img is not None: cv2.imshow("BIN", bin_img)
                if vis_trk is not None: cv2.imshow("Tracker", vis_trk)
            else:
                try: 
                    cv2.destroyWindow("DETECTOR")
                    cv2.destroyWindow("BIN")
                    cv2.destroyWindow("Tracker")
                except: 
                    pass
            
            if cv2.waitKey(1) & 0xFF == ord('q'): break

    except Exception as e:
        print(f"\n主循环异常: {str(e)}")
    except KeyboardInterrupt:
        print("\n收到中断信号...")
    finally:
        print("\n正在释放资源...")
        camera.release()
        uart.close()
        cv2.destroyAllWindows()
        print("系统已安全关闭")

if __name__ == '__main__':
    main()