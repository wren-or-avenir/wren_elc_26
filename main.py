import cv2
import time
import math
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

def nothing(x): pass

def init_board():
    cv2.namedWindow('Controls', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Controls', 400, 600) 
    cv2.namedWindow('DETECTOR', cv2.WINDOW_FREERATIO)  
    cv2.namedWindow('BIN', cv2.WINDOW_FREERATIO)      
    cv2.namedWindow('PROJ', cv2.WINDOW_FREERATIO) 

    # ----- 物理基准与标定 -----
    cv2.createTrackbar('pipe_L', 'Controls', 50, 848, nothing)   # 映射标定：对齐水管真实的左边界
    cv2.createTrackbar('pipe_R', 'Controls', 800, 848, nothing)  # 映射标定：对齐水管真实的右边界
    cv2.createTrackbar('zero_X', 'Controls', 424, 848, nothing)  # 零点标定：对齐水管真实的物理中心X点
    cv2.createTrackbar('zero_Y', 'Controls', 240, 480, nothing)  # 零点标定：控制红色十字的Y轴高度
    cv2.createTrackbar('angle', 'Controls', 5, 10, nothing)      # 倾角补偿：0-10档位对应-5度到+5度(5为0度完全水平)

    # ----- ROI(感兴趣区域)限制 -----
    cv2.createTrackbar('roi_Y1', 'Controls', 200, 480, nothing)  # 图像上边缘：向下压，切除水管上方的画面杂物
    cv2.createTrackbar('roi_Y2', 'Controls', 280, 480, nothing)  # 图像下边缘：向上抬，切除水管下方的画面杂物
    
    # ----- 算法参数 -----
    cv2.createTrackbar('blk_size', 'Controls', 51, 201, nothing) # 二值化块大小：抵抗光照不均，必须为奇数
    cv2.createTrackbar('C_val', 'Controls', 15, 100, nothing)    # 二值化补偿：控制阴影过滤的强度
    cv2.createTrackbar('proj_th', 'Controls', 10, 160, nothing)  # 投影阈值：只有投影峰值高于此数，才判定识别到了钢球
    cv2.createTrackbar('show', 'Controls', 1, 1, nothing)        # 视窗开关：1为显示所有画面，0为关闭画面全速跑

def update_params():
    global show_windows
    show_windows = cv2.getTrackbarPos('show', 'Controls')
    
    pipe_left = cv2.getTrackbarPos('pipe_L', 'Controls')
    pipe_right = cv2.getTrackbarPos('pipe_R', 'Controls')
    tracker.cm_per_pixel = 25.0 / max(1, abs(pipe_right - pipe_left))
    
    zero_x = cv2.getTrackbarPos('zero_X', 'Controls')
    zero_y = cv2.getTrackbarPos('zero_Y', 'Controls')
    angle_deg = cv2.getTrackbarPos('angle', 'Controls') - 5  
    
    ry1 = cv2.getTrackbarPos('roi_Y1', 'Controls')
    ry2 = cv2.getTrackbarPos('roi_Y2', 'Controls')

    block_size = max(3, cv2.getTrackbarPos('blk_size', 'Controls'))
    if block_size % 2 == 0: block_size += 1
    
    c_val = cv2.getTrackbarPos('C_val', 'Controls')
    proj_min_val = cv2.getTrackbarPos('proj_th', 'Controls')
    
    return ry1, ry2, block_size, c_val, proj_min_val, pipe_left, pipe_right, zero_x, zero_y, angle_deg

def main():
    global show_windows
    print("视觉平衡球系统启动... 按 'q' 键退出。")
    init_board()
    
    last_tick = cv2.getTickCount()
    freq = cv2.getTickFrequency()
    
    print_counter = 0
    last_x_offset = 0.0
    last_x_vel = 0.0
    send_hz = 100
    last_send = time.time()

    try:
        while True:
            ret, frame = camera.read()
            if not ret: continue
            
            ry1, ry2, block_size, c_val, proj_min_val, pipe_left, pipe_right, zero_x, zero_y, angle_deg = update_params()

            current_tick = cv2.getTickCount()
            dt = (current_tick - last_tick) / freq
            dt = min(dt, 0.05)  
            last_tick = current_tick
            fps = 1.0 / dt if dt > 0 else 0

            ball_pos = detector.detect(frame, ry1, ry2, block_size, c_val, proj_min_val)
            x_offset, x_vel, status = tracker.track(ball_pos, dt, zero_x)

            if status in [Status.TRACK, Status.TMP_LOST]:
                send_status = 1
                last_x_offset = x_offset
                last_x_vel = x_vel
            else:
                send_status = 0
                x_offset = last_x_offset
                x_vel = last_x_vel
            
            now = time.time()
            if now - last_send >= 1.0 / send_hz:
                cos_theta = math.cos(math.radians(angle_deg))
                if cos_theta <= 0: 
                    cos_theta = 1.0 
                    
                # 物理换算：斜边投影补偿与单位转换
                send_offset_mm = (x_offset / cos_theta) * 10.0
                send_vel_ms = (x_vel / cos_theta) / 100.0
                
                uart.send_data(send_offset_mm, send_vel_ms, send_status)
                last_send = now
            
            print_counter += 1
            if print_counter >= 40:
                info = ""
                if status == Status.TRACK:
                    info = f"[TRACK] dx:{send_offset_mm:>7.2f}mm vx:{send_vel_ms:>6.3f}m/s 状态:{send_status}"
                elif status == Status.TMP_LOST:
                    info = f"[PRED]  dx:{send_offset_mm:>7.2f}mm vx:{send_vel_ms:>6.3f}m/s 状态:{send_status}"
                else:
                    info = f"[LOST]  dx:{send_offset_mm:>7.2f}mm vx:{send_vel_ms:>6.3f}m/s 状态:{send_status}"
                print(f"FPS: {fps:.1f} | {info}")
                print_counter = 0

            if show_windows == 1:
                detector.raw = frame
                vis_det, bin_img, proj_canvas = detector.display(dis=1)
                if vis_det is not None:
                    # 比例标定基准线
                    cv2.line(vis_det, (pipe_left, 0), (pipe_left, 480), (0, 255, 255), 1)
                    cv2.line(vis_det, (pipe_right, 0), (pipe_right, 480), (0, 255, 255), 1)
                    
                    # 红色零点标定十字
                    cv2.line(vis_det, (zero_x, 0), (zero_x, 480), (0, 0, 255), 1)
                    cv2.line(vis_det, (0, zero_y), (848, zero_y), (0, 0, 255), 1)
                    cv2.circle(vis_det, (zero_x, zero_y), 5, (0, 0, 255), -1)
                    
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