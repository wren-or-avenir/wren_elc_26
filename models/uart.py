import serial
import struct

class UartDev:
    def __init__(self, port, baudrate=115200):
        self.port = port
        self.baudrate = baudrate
        self.ser = None
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=0.1)
        except Exception as e:
            print(f"串口打开失败: {e}")

    def send_data(self, offset, vel, status):
        # 协议: 0x55 0xAA(帧头) + 偏移(4B) + 速度(4B) + 状态(1B) + 校验和(1B) + 0x0A(帧尾)
        if self.ser is None or not self.ser.is_open:
            return

        # '<ffB' : 小端序 float, float, unsigned char
        data_bytes = struct.pack('<ffB', float(offset), float(vel), int(status))
        
        # 8位累加和
        checksum = sum(data_bytes) & 0xFF
        
        frame = bytearray([0x55, 0xAA])
        frame.extend(data_bytes)
        frame.append(checksum)
        frame.append(0x0A)
        
        try:
            self.ser.write(frame)
        except Exception as e:
            pass # 遵循无错误处理情景规则，如果写失败忽略，不阻塞主线程

    def close(self):
        if self.ser and self.ser.is_open:
            self.ser.close()