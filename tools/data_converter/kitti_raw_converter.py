import mmcv
import numpy as np
from collections import OrderedDict
from nuscenes.utils.geometry_utils import view_points
from pathlib import Path

from skimage import io 
from mmdet3d.core.bbox import box_np_ops
from tools.data_converter.kitti_data_utils import get_image_index_str
from .nuscenes_converter import post_process_coords
import re 
from concurrent import futures as futures 
import pickle 
from tqdm import tqdm 


kitti_categories = ('Pedestrian', 'Cyclist', 'Car', 'Van', 'Truck')
DATA_CATEGORIES = ('Campus', 'City', 'Person', 'Residential', 'Road')


def create_kitti_raw_info_file(
    data_path:str,
    pkl_prefix='kitti',
    save_path=None, 
    relative_path=True, 
):
    """
    Create info file of KITTI Tracking dataset. 
    """
    with open(str(Path(data_path) / 'kitti_raw_split_train.pkl'), 'rb') as f:
        train_img_ids = pickle.load(f)
    # with open(str(data_path / 'kitti_raw_split_val.pkl'), 'rb') as f: 
    #     val_img_ids = pickle.load(f)
        
    # print(train_img_ids)
    print('{} images in total'.format(len(train_img_ids)))

    # print('Generate info. this may take several seconds.')    
    if save_path is None: 
        save_path = Path(data_path)
    else:
        save_path = Path(save_path)
    kitti_infos_train = get_kitti_raw_image_info(
        data_path, 
        image_ids=train_img_ids,
        training=True, 
        velodyne=True, 
        calib=True, 
        relative_path=relative_path
    )
    filename = save_path / f'{pkl_prefix}_infos_train.pkl'
    print(f'Kitti info train file is saved to {filename}')
    mmcv.dump(kitti_infos_train, filename)


def get_all_image_ids(data_path):
    data_path = Path(data_path)
    for cat in DATA_CATEGORIES:
        cat_dir = data_path / cat.lower()
        if not cat_dir.is_dir():
            raise RuntimeError("Directory {} not exists".format(cat_dir)) 
        
        for seq_dir in cat_dir.iterdir():
            print('seq: ', seq_dir.stem)
            date, seq_id = re.split('_drive_', seq_dir.stem)
            img02_dir = seq_dir / date / f'{date}_drive_{seq_id}_sync' / 'image_02' / 'data'
            if not img02_dir.is_dir():
                # print("Warning: directory {} not exists".format(img02_dir))
                raise RuntimeError("directory {} not exists".format(img02_dir))
                continue 
            for img in img02_dir.iterdir():
                assert img.suffix == '.png', img
                
                yield dict(
                    seq_dir=str(seq_dir.absolute()),
                    date=date, 
                    seq_id=seq_id,
                    frame_id=img.stem,
                    image02_path=str(img.absolute()),
                    image02_path_relative=str(img.relative_to(data_path))
                )


def get_image_index_str(img_info):
    return '{}_{}'.format(img_info['seq_id'], img_info['frame_id'])


def get_image_path(info, data_root, training, relative_path:bool):
    return info['image02_path' if not relative_path else 'image02_path_relative']


def get_velodyne_path(info, data_root, training, relative_path:bool):
    velodyne_path = Path(info['seq_dir']) / info['date'] / '{}_drive_{}_sync'.format(info['date'], info['seq_id']) / 'velodyne_points' / 'data' / '{}.bin'.format(info['frame_id'])
    # assert velodyne_path.is_file(), velodyne_path
    if relative_path:
        return str(velodyne_path.relative_to(data_root.absolute()))
    return str(velodyne_path)
    


def read_img_calib(info):
    def expand_to_4x4(arr):
        rows, cols = arr.shape 
        assert rows <= 4 and cols <= 4 
        ret = np.eye(4) 
        ret[:rows, :cols] = arr 
        return ret 

    img_file = Path(info['image02_path'])
    calib_dir = img_file.parent.parent.parent.parent
    calib_cam_to_cam = calib_dir / 'calib_cam_to_cam.txt'
    # imu has not been supported yet 
    # calib_imu_to_velo = img_file.parent.parent / 'calib_imu_to_velo.txt'
    calib_velo_to_cam = calib_dir / 'calib_velo_to_cam.txt'
    with open(calib_cam_to_cam.absolute(), 'r') as f:
        lines_cam_to_cam = list(f.readlines())
    with open(calib_velo_to_cam.absolute(), 'r') as f: 
        lines_velo_to_cam = list(f.readlines())
    calib = {}
    for line in lines_cam_to_cam:
        key, value = re.split(': ', line)
        if key == 'R_rect_00':
            try: 
                calib['R0_rect'] = np.fromstring(value, sep=' ').reshape((3, 3))
            except Exception as e: 
                print("file: {}".format(calib_cam_to_cam))
                print("lines: {}".format(line)) 
                print("value: {}".format(value)) 
                raise e 
        elif key == 'P_rect_02':
            calib['P2'] = np.fromstring(value, sep=' ').reshape((3, 4))

    Tr_velo_to_cam = np.eye(4)
    for line in lines_velo_to_cam:
        key, value = re.split(': ', line) 
        if key == 'R': 
            Tr_velo_to_cam[:3, :3] = np.fromstring(value, sep=' ').reshape((3, 3))
        elif key == 'T': 
            Tr_velo_to_cam[:3, -1] = np.fromstring(value, sep=' ')
    calib['Tr_velo_to_cam'] = Tr_velo_to_cam
        
    calib = {k: expand_to_4x4(v) for k, v in calib.items()}
    return calib 


def get_kitti_raw_image_info(
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

    def for_each_image(frame_info):
        sequence_str = frame_info['seq_id']
        frame_str = frame_info['frame_id']
        img02_path = frame_info['image02_path']

        info = {}
        calib_info = {}

        image_info = {
            'image_idx': get_image_index_str(frame_info)
        }
        image_info['image_path'] = get_image_path(frame_info, root_path, training, relative_path)
        # print(img_info)(img_info, path, training, relative_path)
        if with_imageshape:
            img_path = image_info['image_path']
            if relative_path:
                img_path = str(root_path / img_path)
            image_info['image_shape'] = np.array(
                io.imread(img_path).shape[:2], dtype=np.int32
            )
        pc_info = {'num_features': 4}
        pc_info['velodyne_path'] = get_velodyne_path(frame_info, root_path, training, relative_path)
        info['image'] = image_info
        info['point_cloud'] = pc_info 
        if calib: 
            info['calib'] = read_img_calib(frame_info)
        
        return info         
    
    if num_worker > 1:
        with futures.ThreadPoolExecutor(num_worker) as executor:
            image_infos = executor.map(for_each_image, image_ids)
    else:
        image_infos = [for_each_image(i) for i in tqdm(image_ids)]
    
    return list(image_infos)