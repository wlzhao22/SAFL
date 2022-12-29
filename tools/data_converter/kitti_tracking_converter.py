import mmcv
import numpy as np
from collections import OrderedDict
from nuscenes.utils.geometry_utils import view_points
from pathlib import Path

from mmdet3d.core.bbox import box_np_ops
from .kitti_tracking_data_utils import get_kitti_tracking_image_info
from .nuscenes_converter import post_process_coords
from tools.data_converter.kitti_tracking_splits import AB3DMOTSplit, CenterTrackSplit, SPLIT_REGISTRY


kitti_categories = ('Pedestrian', 'Cyclist', 'Car')

def create_kitti_tracking_info_file(
    data_path:str,
    pkl_prefix='kitti',
    save_path=None, 
    relative_path=True, 
    splits='ab3dmot'
):
    """
    Create info file of KITTI Tracking dataset. 
    """
    splits = SPLIT_REGISTRY.get(splits)
    train_img_ids = list(splits.train_frameset_ids(root=data_path))
    val_img_ids = list(splits.val_frameset_ids(root=data_path))
    test_img_ids = list(splits.test_frameset_ids(root=data_path))

    print('Generate info. this may take several seconds.')    
    if save_path is None: 
        save_path = Path(data_path)
    else:
        save_path = Path(save_path)
    kitti_infos_train = get_kitti_tracking_image_info(
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
    kitti_infos_val = get_kitti_tracking_image_info(
        data_path,
        training=True,
        velodyne=True,
        calib=True,
        image_ids=val_img_ids,
        relative_path=relative_path)
    filename = save_path / f'{pkl_prefix}_infos_val.pkl'
    print(f'Kitti info val file is saved to {filename}')
    mmcv.dump(kitti_infos_val, filename)
    filename = save_path / f'{pkl_prefix}_infos_trainval.pkl'
    print(f'Kitti info trainval file is saved to {filename}')
    mmcv.dump(kitti_infos_train + kitti_infos_val, filename)
    
    kitti_infos_test = get_kitti_tracking_image_info(
        data_path,
        training=False,
        label_info=False,
        velodyne=True,
        calib=True,
        image_ids=test_img_ids,
        relative_path=relative_path)
    filename = save_path / f'{pkl_prefix}_infos_test.pkl'
    print(f'Kitti info test file is saved to {filename}')
    mmcv.dump(kitti_infos_test, filename)



def get_2d_boxes(info, occluded, mono3d=True):
    """Get the 2D annotation records for a given info.

    Args:
        info: Information of the given sample data.
        occluded: Integer (0, 1, 2, 3) indicating occlusion state: \
            0 = fully visible, 1 = partly occluded, 2 = largely occluded, \
            3 = unknown, -1 = DontCare
        mono3d (bool): Whether to get boxes with mono3d annotation.

    Return:
        list[dict]: List of 2D annotation record that belongs to the input
            `sample_data_token`.
    """
    # Get calibration information
    P2 = info['calib']['P2']

    repro_recs = []
    # if no annotations in info (test dataset), then return
    if 'annos' not in info:
        return repro_recs

    # Get all the annotation with the specified visibilties.
    ann_dicts = info['annos']
    mask = [(ocld in occluded) for ocld in ann_dicts['occluded']]
    for k in ann_dicts.keys():
        ann_dicts[k] = ann_dicts[k][mask]

    # convert dict of list to list of dict
    ann_recs = []
    for i in range(len(ann_dicts['occluded'])):
        ann_rec = {}
        for k in ann_dicts.keys():
            ann_rec[k] = ann_dicts[k][i]
        ann_recs.append(ann_rec)

    for ann_idx, ann_rec in enumerate(ann_recs):
        # Augment sample_annotation with token information.
        ann_rec['sample_annotation_token'] = \
            f"{info['image']['image_idx']}.{ann_idx}"
        ann_rec['sample_data_token'] = info['image']['image_idx']
        sample_data_token = info['image']['image_idx']

        loc = ann_rec['location'][np.newaxis, :]
        dim = ann_rec['dimensions'][np.newaxis, :]
        rot = ann_rec['rotation_y'][np.newaxis, np.newaxis]
        # transform the center from [0.5, 1.0, 0.5] to [0.5, 0.5, 0.5]
        dst = np.array([0.5, 0.5, 0.5])
        src = np.array([0.5, 1.0, 0.5])
        loc = loc + dim * (dst - src)
        offset = (info['calib']['P2'][0, 3] - info['calib']['P0'][0, 3]) \
            / info['calib']['P2'][0, 0]
        loc_3d = np.copy(loc)
        loc_3d[0, 0] += offset
        gt_bbox_3d = np.concatenate([loc, dim, rot], axis=1).astype(np.float32)

        # Filter out the corners that are not in front of the calibrated
        # sensor.
        corners_3d = box_np_ops.center_to_corner_box3d(
            gt_bbox_3d[:, :3],
            gt_bbox_3d[:, 3:6],
            gt_bbox_3d[:, 6], [0.5, 0.5, 0.5],
            axis=1)
        corners_3d = corners_3d[0].T  # (1, 8, 3) -> (3, 8)
        in_front = np.argwhere(corners_3d[2, :] > 0).flatten()
        corners_3d = corners_3d[:, in_front]

        # Project 3d box to 2d.
        camera_intrinsic = P2
        corner_coords = view_points(corners_3d, camera_intrinsic,
                                    True).T[:, :2].tolist()

        # Keep only corners that fall within the image.
        final_coords = post_process_coords(corner_coords)

        # Skip if the convex hull of the re-projected corners
        # does not intersect the image canvas.
        if final_coords is None:
            continue
        else:
            min_x, min_y, max_x, max_y = final_coords

        # Generate dictionary record to be included in the .json file.
        repro_rec = generate_record(ann_rec, min_x, min_y, max_x, max_y,
                                    sample_data_token,
                                    info['image']['image_path'])

        # If mono3d=True, add 3D annotations in camera coordinates
        if mono3d and (repro_rec is not None):
            repro_rec['bbox_cam3d'] = np.concatenate(
                [loc_3d, dim, rot],
                axis=1).astype(np.float32).squeeze().tolist()
            repro_rec['velo_cam3d'] = -1  # no velocity in KITTI

            center3d = np.array(loc).reshape([1, 3])
            center2d = box_np_ops.points_cam2img(
                center3d, camera_intrinsic, with_depth=True)
            repro_rec['center2d'] = center2d.squeeze().tolist()
            # normalized center2D + depth
            # samples with depth < 0 will be removed
            if repro_rec['center2d'][2] <= 0:
                continue

            repro_rec['attribute_name'] = -1  # no attribute in KITTI
            repro_rec['attribute_id'] = -1

        repro_recs.append(repro_rec)

    return repro_recs


def generate_record(ann_rec, x1, y1, x2, y2, sample_data_token, filename):
    """Generate one 2D annotation record given various informations on top of
    the 2D bounding box coordinates.

    Args:
        ann_rec (dict): Original 3d annotation record.
        x1 (float): Minimum value of the x coordinate.
        y1 (float): Minimum value of the y coordinate.
        x2 (float): Maximum value of the x coordinate.
        y2 (float): Maximum value of the y coordinate.
        sample_data_token (str): Sample data token.
        filename (str):The corresponding image file where the annotation
            is present.

    Returns:
        dict: A sample 2D annotation record.
            - file_name (str): flie name
            - image_id (str): sample data token
            - area (float): 2d box area
            - category_name (str): category name
            - category_id (int): category id
            - bbox (list[float]): left x, top y, dx, dy of 2d box
            - iscrowd (int): whether the area is crowd
    """
    repro_rec = OrderedDict()
    repro_rec['sample_data_token'] = sample_data_token
    coco_rec = dict()

    key_mapping = {
        'name': 'category_name',
        'num_points_in_gt': 'num_lidar_pts',
        'sample_annotation_token': 'sample_annotation_token',
        'sample_data_token': 'sample_data_token',
    }

    for key, value in ann_rec.items():
        if key in key_mapping.keys():
            repro_rec[key_mapping[key]] = value

    repro_rec['bbox_corners'] = [x1, y1, x2, y2]
    repro_rec['filename'] = filename

    coco_rec['file_name'] = filename
    coco_rec['image_id'] = sample_data_token
    coco_rec['area'] = (y2 - y1) * (x2 - x1)

    if repro_rec['category_name'] not in kitti_categories:
        return None
    cat_name = repro_rec['category_name']
    coco_rec['category_name'] = cat_name
    coco_rec['category_id'] = kitti_categories.index(cat_name)
    coco_rec['bbox'] = [x1, y1, x2 - x1, y2 - y1]
    coco_rec['iscrowd'] = 0

    return coco_rec
