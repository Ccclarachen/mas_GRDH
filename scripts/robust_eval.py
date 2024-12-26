"""
This file is writing for testing robustness of our method
"""
from utils import image_grid, load_512
import cv2
import torch
import numpy as np


def _store(samples, tmp_image_name):
    image_grid(samples).save(f'{tmp_image_name}.png')
    # x0_samples = load_512('./tmp.png')
    samples = cv2.imread(f'{tmp_image_name}.png')
    samples = cv2.cvtColor(samples, cv2.COLOR_BGR2RGB)
    return samples


def _store_jpeg(samples, factor, tmp_image_name):
    image_grid(samples).save(f'{tmp_image_name}.jpg', quality=factor)
    # x0_samples = load_512('./tmp.png')
    samples = cv2.imread(f'{tmp_image_name}.jpg')
    samples = cv2.cvtColor(samples, cv2.COLOR_BGR2RGB)
    return samples


# without any lossy operations
def identity(samples, factor=None, tmp_image_name='tmp'):
    return samples


# saving and reloading
def storage(samples, factor=None, tmp_image_name='tmp'):
    samples = _store(samples, tmp_image_name=tmp_image_name)
    reload_samples = load_512(samples)
    return reload_samples


def resize(samples, factor, tmp_image_name='tmp'):
    samples = _store(samples, tmp_image_name=tmp_image_name)
    height, width = samples.shape[:2]
    new_height = int(height * factor)
    new_width = int(width * factor)
    scaled_samples = cv2.resize(samples, (new_width, new_height), interpolation=cv2.INTER_LINEAR)  # 使用双线性插值
    recover_samples = cv2.resize(scaled_samples, (width, height), interpolation=cv2.INTER_LINEAR)  # 使用双线性插值
    reload_samples = load_512(recover_samples)
    return reload_samples


def jpeg(samples, factor, tmp_image_name='tmp'):
    samples = _store_jpeg(samples, int(factor), tmp_image_name=tmp_image_name)
    reload_samples = load_512(samples)
    return reload_samples


def mblur(samples, factor, tmp_image_name='tmp'):
    samples = _store(samples, tmp_image_name=tmp_image_name)
    samples = cv2.medianBlur(samples, int(factor))
    reload_samples = load_512(samples)
    return reload_samples


def gblur(samples, factor, tmp_image_name='tmp'):
    samples = _store(samples, tmp_image_name=tmp_image_name)
    samples = cv2.GaussianBlur(samples, (int(factor), int(factor)), 0)
    reload_samples = load_512(samples)
    return reload_samples


def awgn(samples, factor, tmp_image_name='tmp'):
    samples = _store(samples, tmp_image_name=tmp_image_name)
    samples = load_512(samples)
    samples = samples + torch.tensor(np.random.normal(0, factor, samples.shape)).to(samples)  # 高斯噪声
    reload_samples = torch.clamp(samples, min=-1., max=1.)
    return reload_samples
