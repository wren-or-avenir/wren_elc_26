class KalmanFilter:
    def __init__(self, q_scale=1.0, r_scale=0.2):
        # 放弃 OpenCV 矩阵运算，改为极简稳态增益 (Alpha-Beta 滤波)
        # 保留 q_scale, r_scale 入参仅为兼容原有的 Tracker 接口调用
        self.alpha = 0.65
        self.beta = 0.35
        self.dt = 0.033
        
        # 突变阈值(cm)：观测值与预测值差值超过此值，判定为发生突变/反弹
        self.jump_threshold = 1.0 
        self.is_init = False
        self.reset()

    def predict(self, dt=None):
        if dt is not None:
            self.dt = dt
        if not self.is_init:
            return 0.0, 0.0
        return self.x + self.v * self.dt, self.v

    def update(self, measurement):
        if not self.is_init:
            self.x = measurement
            self.v = 0.0
            self.is_init = True
            return self.x, self.v

        pred_x = self.x + self.v * self.dt
        residual = measurement - pred_x

        # 针对突变滞后的核心解决逻辑：
        # 若残差极大，说明发生急剧变速或反向，彻底放弃滤波惯性，直接采信当前观测值
        if abs(residual) > self.jump_threshold:
            if self.dt > 0:
                self.v = (measurement - self.x) / self.dt
            else:
                self.v = 0.0
            self.x = measurement
        else:
            self.x = pred_x + self.alpha * residual
            if self.dt > 0:
                self.v = self.v + (self.beta / self.dt) * residual

        return self.x, self.v

    def reset(self):
        self.x = 0.0
        self.v = 0.0
        self.is_init = False