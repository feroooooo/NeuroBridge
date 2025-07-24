import numpy as np
import random

class RandomTimeShift:
    """
    在时间维度 (axis=-1) 上进行随机平移。
    max_shift 表示最大平移量（向前或向后），单位是采样点。
    """
    def __init__(self, max_shift=5):
        self.max_shift = max_shift

    def __call__(self, eeg_data: np.ndarray) -> np.ndarray:
        # eeg_data shape: (..., time)
        shift = random.randint(-self.max_shift, self.max_shift)
        if shift != 0:
            eeg_data = np.roll(eeg_data, shift, axis=-1)
        return eeg_data


class RandomGaussianNoise:
    """
    在 EEG 信号中添加随机高斯噪声。
    std 表示噪声标准差。
    """
    def __init__(self, std=0.01):
        self.std = std

    def __call__(self, eeg_data: np.ndarray) -> np.ndarray:
        noise = np.random.normal(0, self.std, size=eeg_data.shape)
        return eeg_data + noise


class RandomChannelDropout:
    """
    随机丢弃一部分通道（置为 0）。
    drop_prob 表示丢弃某个通道的概率。
    假设 eeg_data shape: (channel, time) 或 (channel, time, ...)。
    """
    def __init__(self, drop_prob=0.1):
        self.drop_prob = drop_prob

    def __call__(self, eeg_data: np.ndarray) -> np.ndarray:
        # 假设第一个维度是通道维度
        channels = eeg_data.shape[0]
        for ch in range(channels):
            # 以 drop_prob 的概率将该通道置零
            if random.random() < self.drop_prob:
                eeg_data[ch] = 0
        return eeg_data


class RandomSmooth:
    """
    简单的平滑操作，可理解为在时间轴上做一个简单卷积 / 移动平均。
    kernel_size 表示移动平均核的大小
    """
    def __init__(self, kernel_size=5, smooth_prob=0.5):
        self.kernel_size = kernel_size
        self.smooth_prob = smooth_prob

    def __call__(self, eeg_data: np.ndarray) -> np.ndarray:
        ch, time_len = eeg_data.shape
        smoothed = np.copy(eeg_data)
        # 对每个通道在时间维度做移动平均
        for c in range(ch):
            if np.random.rand() < self.smooth_prob:
                for t in range(time_len):
                    left = max(0, t - self.kernel_size // 2)
                    right = min(time_len, t + self.kernel_size // 2 + 1)
                    smoothed[c, t] = np.mean(eeg_data[c, left:right])
        return smoothed



class RandomApply:
    """
    给定一个 transform(增强)，以概率 p 随机执行该 transform， 否则不执行。
    """
    def __init__(self, transform, p=0.5):
        self.transform = transform
        self.p = p

    def __call__(self, eeg_data: np.ndarray) -> np.ndarray:
        if random.random() < self.p:
            return self.transform(eeg_data)
        return eeg_data