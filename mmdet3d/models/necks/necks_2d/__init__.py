'''
Author: your name
Date: 2021-04-02 10:08:57
LastEditTime: 2021-04-02 10:19:27
LastEditors: your name
Description: In User Settings Edit
FilePath: /BDPilot_RepVGG_v3/mmdet/models/bdnet_necks/__init__.py
'''
from .yolo_neck import YoloNeck
from .bifpn import BiFPN

__all__ = [
    'YoloNeck', 'BiFPN'
]
