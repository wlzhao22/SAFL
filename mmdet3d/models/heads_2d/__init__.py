'''
Author: your name
Date: 2020-12-06 21:03:32
LastEditTime: 2021-06-22 16:05:06
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /BDPilot/mmdet/models/bdnet/__init__.py
'''
from .train.depth_head import DepthHead
from .train.detect_3d_head import Detect3DHead
from .train.lane_head import LaneHead
from .train.pose_head import PoseHead
from .train.auto_head import AutoHead
from .train.detect_head import DetectHead
from .train.freespace_head import FreeSpaceHead
from .train.det_head import DetHead
from .train.yolov5 import YOLOv5Head
from .train.centerhead_with_fpn import CenterNetFPNHead
from .train.centernet_head import CenterNetHead
from .train.keypoint_head import KeyPointHead
from .train.lane_freespace_head import LaneFreespaceHead
from .distill.lane_head_teacher import LaneHeadTea
from .distill.freespace_head_teacher import FreeSpaceHeadTea
from .distill.yolov5_teacher import YOLOv5HeadTea
from .distill.centernet_head_teacher import CenterNetHeadTea
from .distill.lane_head_student import LaneHeadStu
from .distill.freespace_head_student import FreeSpaceHeadStu
from .distill.yolov5_student import YOLOv5HeadStu
from .distill.centernet_head_student import CenterNetHeadStu
from .distill.track_head_teacher import TrackHeadTea
from .distill.track_head_student import TrackHeadStu
from .prune.lane_head_prune import LaneHeadPrune

__all__ = [
    'DepthHead', 'DetectHead', 'LaneHead', 'PoseHead', 'AutoHead','Detect3DHead','FreeSpaceHead','DetHead', \
    'YOLOv5Head', 'CenterNetHead', 'KeyPointHead', 'LaneFreespaceHead','LaneHeadTea','FreeSpaceHeadTea','YOLOv5HeadTea', \
    'CenterNetHeadTea','LaneHeadStu','FreeSpaceHeadStu','YOLOv5HeadStu','CenterNetHeadStu','TrackHeadTea','TrackHeadStu', 'LaneHeadPrune'
]
