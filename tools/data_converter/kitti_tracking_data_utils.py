
from pathlib import Path
from tools.data_converter.kitti_data_utils import add_difficulty_to_annos
from typing import Union
from collections import defaultdict 
from concurrent import futures as futures 
from pathlib import Path 
import numpy as np 
from skimage import io 


def get_image_index_str(img_idx):
    if img_idx['sequence_id'] is not None: 
        return "{}/{}".format(img_idx['sequence_id'], img_idx['frame_id'])
    else:
        return img_idx['frame_id']


def get_kitti_info_path(
    idx,
    prefix,
    info_type="image_02",
    file_tail=".png",
    training=True,
    relative_path=True,
    exist_check=True,
):
    img_idx_str = get_image_index_str(idx)
    img_idx_str += file_tail
    prefix = Path(prefix)
    if training:
        file_path = Path("training") / info_type / img_idx_str
    else:
        file_path = Path("testing") / info_type / img_idx_str
    if not (prefix / file_path).exists():
        if exist_check:
            raise ValueError("file not exist: {}".format(prefix / file_path))
        else:
            return ''
    if relative_path:
        return str(file_path)
    else:
        return str(prefix / file_path)


def get_velodyne_path(idx, prefix, training=True, relative_path=True, exist_check=False):
    return get_kitti_info_path(
        idx, prefix, "velodyne", ".bin", training, relative_path, exist_check
    )


def get_image_path(idx, prefix, training=True, relative_path=True, exist_check=True):
    return get_kitti_info_path(
        idx, prefix, "image_02", ".png", training, relative_path, exist_check
    )


def get_label_path(idx, prefix, training=True, relative_path=True, exist_check=True):
    label_idx_str = idx['sequence_id'] if idx['sequence_id'] is not None else idx['frame_id']
    label_idx_str += ".txt"
    prefix = Path(prefix)
    if training:
        file_path = Path("training") / "label_02" / label_idx_str
    else:
        file_path = Path("testing") / "label_02" / label_idx_str
    if exist_check and not (prefix / file_path).exists():
        raise ValueError("file not exist: {}".format(prefix / file_path))
    if relative_path:
        return str(file_path)
    else:
        return str(prefix / file_path)


def get_label_anno(label_path):
    annotations = {}
    annotations.update(
        {
            "name": [],
            "truncated": [],
            "occluded": [],
            "alpha": [],
            "bbox": [],
            "dimensions": [],
            "location": [],
            "rotation_y": [],
        }
    )
    with open(label_path, "r") as f:
        lines = f.readlines()
    # if len(lines) == 0 or len(lines[0]) < 15:
    #     content = []
    # else:
    content = [line.strip().split(" ") for line in lines]
    num_objects = len([x[0] for x in content if x[0] != "DontCare"])
    annotations['frame_id'] = np.array([x[0] for x in content])
    annotations['track_id'] = np.array([x[1] for x in content])
    annotations["name"] = np.array([x[2] for x in content])
    num_gt = len(annotations["name"])
    annotations["truncated"] = np.array([float(x[3]) for x in content])
    annotations["occluded"] = np.array([int(x[4]) for x in content])
    annotations["alpha"] = np.array([float(x[5]) for x in content])
    annotations["bbox"] = np.array(
        [[float(info) for info in x[6:10]] for x in content]
    ).reshape(-1, 4)
    # dimensions will convert hwl format to standard lhw(camera) format.
    annotations["dimensions"] = np.array(
        [[float(info) for info in x[10:13]] for x in content]
    ).reshape(-1, 3)[:, [2, 0, 1]]
    annotations["location"] = np.array(
        [[float(info) for info in x[13:16]] for x in content]
    ).reshape(-1, 3)
    annotations["rotation_y"] = np.array([float(x[16]) for x in content]).reshape(-1)
    if len(content) != 0 and len(content[0]) == 18:  # have score
        annotations["score"] = np.array([float(x[17]) for x in content])
    # else:
    #     annotations["score"] = np.zeros((annotations["bbox"].shape[0],))
    index = []
    flag = 0
    for name in annotations['name']:
        if name == 'DontCare': index.append(-1)
        else: index.append(flag); flag += 1
    assert len(index) == num_gt
    annotations["index"] = np.array(index, dtype=np.int32)
    annotations["group_ids"] = np.arange(num_gt, dtype=np.int32)
    return annotations


def get_calib_path(idx, prefix, training=True, relative_path=True, exist_check=True):
    img_idx_str = idx['sequence_id']
    img_idx_str += ".txt"
    prefix = Path(prefix)
    if training:
        file_path = Path("training") / "calib" / img_idx_str
    else:
        file_path = Path("testing") / "calib" / img_idx_str
    if exist_check and not (prefix / file_path).exists():
        raise ValueError("file not exist: {}".format(prefix / file_path))
    if relative_path:
        return str(file_path)
    else:
        return str(prefix / file_path)


def _extend_matrix(mat):
    mat = np.concatenate([mat, np.array([[0.0, 0.0, 0.0, 1.0]])], axis=0)
    return mat


def for_each_sequence(
    sequence_id:str, root_path:Path, training:bool, label_info:bool, 
    velodyne:bool, calib:bool, oxts:bool, extend_matrix:bool, 
    relative_path
):
    """
    For each sequence, find its label file path, and load the oxts data, calibration data. 
    """
    annotations = None 
    calib_info = {}
    idx = {'sequence_id': sequence_id}
    if label_info:
        label_path = get_label_path(idx, root_path, training, relative_path)
        if relative_path:
            label_path = str(root_path / label_path)
        annotations = get_label_anno(label_path)

    if calib:
        calib_path = get_calib_path(idx, root_path, training, relative_path=False)
        with open(calib_path, "r") as f:
            lines = f.readlines()
        P0 = np.array([float(info) for info in lines[0].split(" ")[1:13]]).reshape(
            [3, 4]
        )
        P1 = np.array([float(info) for info in lines[1].split(" ")[1:13]]).reshape(
            [3, 4]
        )
        P2 = np.array([float(info) for info in lines[2].split(" ")[1:13]]).reshape(
            [3, 4]
        )
        P3 = np.array([float(info) for info in lines[3].split(" ")[1:13]]).reshape(
            [3, 4]
        )
        if extend_matrix:
            P0 = _extend_matrix(P0)
            P1 = _extend_matrix(P1)
            P2 = _extend_matrix(P2)
            P3 = _extend_matrix(P3)
        R0_rect = np.array(
            [float(info) for info in lines[4].split(" ")[1:10]]
        ).reshape([3, 3])
        if extend_matrix:
            rect_4x4 = np.zeros([4, 4], dtype=R0_rect.dtype)
            rect_4x4[3, 3] = 1.0
            rect_4x4[:3, :3] = R0_rect
        else:
            rect_4x4 = R0_rect

        Tr_velo_to_cam = np.array(
            [float(info) for info in lines[5].split(" ")[1:13]]
        ).reshape([3, 4])
        Tr_imu_to_velo = np.array(
            [float(info) for info in lines[6].split(" ")[1:13]]
        ).reshape([3, 4])
        if extend_matrix:
            Tr_velo_to_cam = _extend_matrix(Tr_velo_to_cam)
            Tr_imu_to_velo = _extend_matrix(Tr_imu_to_velo)
        calib_info["P0"] = P0
        calib_info["P1"] = P1
        calib_info["P2"] = P2
        calib_info["P3"] = P3
        calib_info["R0_rect"] = rect_4x4
        calib_info["Tr_velo_to_cam"] = Tr_velo_to_cam
        calib_info["Tr_imu_to_velo"] = Tr_imu_to_velo

    if oxts: 
        oxts_path = get_oxts_path(idx, path, training, relative_path)
        oxts_data = read_oxts_file(path / oxts_path)
        list_poses = list(convertOxtsToPose(oxts_data))  # IMU poses for this sequence
    else:
        oxts_data = None
        list_poses = None 
    return annotations, calib_info, oxts_data, list_poses


def get_kitti_tracking_image_info(
    path, 
    image_ids,
    training=True, 
    label_info=True, 
    velodyne=False, 
    calib=False, 
    extend_matrix=True, 
    num_worker=8,
    relative_path=True,
    with_imageshape=True, 
):
    root_path = Path(path)

    sequences:set = set([i['sequence_id'] for i in image_ids])
    sequence_infos = {seq: for_each_sequence(
            seq, root_path=root_path, training=training, 
            label_info=label_info, velodyne=velodyne, calib=calib, oxts=False, extend_matrix=extend_matrix,
            relative_path=relative_path
        ) for seq in sequences
    }

    def map_func(idx):
        sequence_str = idx['sequence_id']
        frame_str = idx['frame_id']

        sequence_annos, sequence_calib, sequence_oxts_data, sequence_poses = sequence_infos[sequence_str]

        info = {}
        pc_info = {'num_features': 4}
        calib_info = {}

        image_info = {'image_idx': get_image_index_str(idx)}
        annotations = None 
        if velodyne:
            pc_info['velodyne_path'] = get_velodyne_path(
                idx, path, training, relative_path, exist_check=False
            )
        image_info['image_path'] = get_image_path(idx, path, training, relative_path)
        if with_imageshape:
            img_path = image_info['image_path']
            if relative_path:
                img_path = str(root_path / img_path)
            image_info['image_shape'] = np.array(
                io.imread(img_path).shape[:2], dtype=np.int32
            )
        if label_info:
            frame_annos_ids = [j for j in range(sequence_annos['frame_id'].shape[0]) if int(sequence_annos['frame_id'][j]) == int(frame_str)] \
                if sequence_annos is not None else None 
            annotations = {k: sequence_annos[k][frame_annos_ids] for k in sequence_annos} \
                if frame_annos_ids is not None else None
        info['image'] = image_info
        info['point_cloud'] = pc_info 
        if calib: 
            info['calib'] = sequence_calib
        
        if annotations is not None: 
            info['annos'] = annotations 
            add_difficulty_to_annos(info)
        return info         
    
    if num_worker > 1:
        with futures.ThreadPoolExecutor(num_worker) as executor:
            image_infos = executor.map(map_func, image_ids)
    else:
        image_infos = [map_func(i) for i in image_ids]
    
    return list(image_infos)