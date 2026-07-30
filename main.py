import cv2
import time
from models.cam import Camera
from models.detector import Detector
from models.tracker import Tracker, Status
from models.uart import UartDev

camera_index = 4             
uart_port = '/dev/ttyACM0'      
use_kf = True           
show_windows = 1     

camera = Camera(index=camera_index, width=848, height=480)
detector = Detector(img_width=848, img_height=480)
tracker = Tracker(img_width=848, use_kf=use_kf) 
uart = UartDev(port=uart_port, baudrate=115200)

brake_th = 1.0 

def nothing(x): pass

def init_board():
    cv2.namedWindow('Controls', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Controls', 300, 300) 
    cv2.namedWindow('DETECTOR', cv2.WINDOW_FREERATIO)  
    cv2.namedWindow('BIN', cv2.WINDOW_FREERATIO)      
    cv2.namedWindow('PROJ', cv2.WINDOW_FREERATIO) 

    cv2.createTrackbar('brake_th', 'Controls', 10, 100, nothing) 
    cv2.createTrackbar('pipe_L', 'Controls', 50, 848, nothing) 
    cv2.createTrackbar('pipe_R', 'Controls', 800, 848, nothing) 
    
    cv2.createTrackbar('roi_h', 'Controls', 100, 480, nothing) 
    cv2.createTrackbar('blk_size', 'Controls', 51, 201, nothing) 
    cv2.createTrackbar('C_val', 'Controls', 15, 100, nothing) 
    cv2.createTrackbar('proj_th', 'Controls', 10, 160, nothing) 
    cv2.createTrackbar('show', 'Controls', 1, 1, nothing)

def update_params():
    global brake_th, show_windows
    brake_th = cv2.getTrackbarPos('brake_th', 'Controls') / 10.0
    show_windows = cv2.getTrackbarPos('show', 'Controls')
    
    pipe_left = cv2.getTrackbarPos('pipe_L', 'Controls')
    pipe_right = cv2.getTrackbarPos('pipe_R', 'Controls')
    tracker.cm_per_pixel = 25.0 / max(1, abs(pipe_right - pipe_left))
    
    roi_height = max(10, cv2.getTrackbarPos('roi_h', 'Controls'))
    block_size = max(3, cv2.getTrackbarPos('blk_size', 'Controls'))
    if block_size % 2 == 0: block_size += 1
    
    c_val = cv2.getTrackbarPos('C_val', 'Controls')
    proj_min_val = cv2.getTrackbarPos('proj_th', 'Controls')
    return roi_height, block_size, c_val, proj_min_val, pipe_left, pipe_right

def main():
    global show_windows, brake_th
    print("视觉平衡球系统启动... 按 'q' 键退出。")
    init_board()
    
    last_tick = cv2.getTickCount()
    freq = cv2.getTickFrequency()
    
    print_counter = 0
    last_x_offset = 0.0
    last_x_vel = 0.0

    try:
        while True:
            ret, frame = camera.read()
            if not ret: continue
            
            current_tick = cv2.getTickCount()
            dt = (current_tick - last_tick) / freq
            dt = min(dt, 0.05)  
            last_tick = current_tick
            fps = 1.0 / dt if dt > 0 else 0

            roi_height, block_size, c_val, proj_min_val, pipe_left, pipe_right = update_params()

            ball_pos = detector.detect(frame, roi_height, block_size, c_val, proj_min_val)
            
            x_offset, x_vel, status = tracker.track(ball_pos, dt)

            # 直接通过 if-else 将内部状态强转为 0 和 1
            if status in [Status.TRACK, Status.TMP_LOST]:
                send_status = 1
                # 记录有效状态下数据供丢失时使用
                last_x_offset = x_offset
                last_x_vel = x_vel
            else:
                send_status = 0
                # 取出历史预测缓存继续下发
                x_offset = last_x_offset
                x_vel = last_x_vel
            
            uart.send_data(x_offset, x_vel, send_status)
            
            print_counter += 1
            if print_counter >= 40:
                info = ""
                if status == Status.TRACK:
                    info = f"[TRACK] dx:{x_offset:>6.2f}cm vx:{x_vel:>6.2f}cm/s 状态:{send_status}"
                elif status == Status.TMP_LOST:
                    info = f"[PRED]  dx:{x_offset:>6.2f}cm vx:{x_vel:>6.2f}cm/s 状态:{send_status}"
                else:
                    info = f"[LOST]  dx:{x_offset:>6.2f}cm vx:{x_vel:>6.2f}cm/s 状态:{send_status}"
                print(f"FPS: {fps:.1f} | {info}")
                print_counter = 0

            if show_windows == 1:
                detector.raw = frame
                vis_det, bin_img, proj_canvas = detector.display(dis=1)
                if vis_det is not None:
                    cv2.line(vis_det, (pipe_left, 0), (pipe_left, 480), (0, 255, 255), 1)
                    cv2.line(vis_det, (pipe_right, 0), (pipe_right, 480), (0, 255, 255), 1)
                    cv2.line(vis_det, (424, 0), (424, 480), (0, 165, 255), 1)
                    cv2.circle(vis_det, (424, 240), 5, (0, 165, 255), -1)
                    cv2.imshow("DETECTOR", vis_det)
                if bin_img is not None: 
                    cv2.imshow("BIN", bin_img)
                if proj_canvas is not None:
                    cv2.imshow("PROJ", proj_canvas)
            else:
                try: 
                    cv2.destroyWindow("DETECTOR")
                    cv2.destroyWindow("BIN")
                    cv2.destroyWindow("PROJ")
                except: pass
            
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