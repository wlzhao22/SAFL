from pathlib import Path 
from typing import Union
from collections import defaultdict 


class SplitRegistry:
    def __init__(self):
        self.modules = dict() 

    def register(self, Cls):
        self.modules[Cls.name] = Cls

    def get(self, name):
        return self.modules[name]
SPLIT_REGISTRY = SplitRegistry()


@SPLIT_REGISTRY.register
class CenterTrackSplit:
    name='centertrack'
    @staticmethod
    def train_frameset_ids(root:Union[Path, str]):
        if isinstance(root, str):
            root = Path(root)
        root_train = root / 'training'
        root_image_02 = root_train / 'image_02'
        for sequence_path in sorted(root_image_02.iterdir()):
            sequence_id = sequence_path.stem 
            image_path_list = sorted(list(sequence_path.iterdir()))
            image_path_list = image_path_list[:(len(image_path_list)//2)]
            for image_path in image_path_list:
                image_id = image_path.stem 
                yield {
                    'sequence_id': sequence_id, 'frame_id': image_id
                }
                
    @staticmethod
    def test_frameset_ids(root:Union[Path, str]):
        if isinstance(root, str):
            root = Path(root)
        root_train = root / 'testing'
        root_image_02 = root_train / 'image_02'
        for sequence_path in sorted(root_image_02.iterdir()):
            sequence_id = sequence_path.stem 
            for image_path in sorted(sequence_path.iterdir()):
                image_id = image_path.stem 
                yield {
                    'sequence_id': sequence_id, 'frame_id': image_id
                }

    @staticmethod
    def val_frameset_ids(root:Union[Path, str]):
        if isinstance(root, str):
            root = Path(root)
        root_train = root / 'training'
        root_image_02 = root_train / 'image_02'
        for sequence_path in sorted(root_image_02.iterdir()):
            sequence_id = sequence_path.stem 
            image_path_list = sorted(list(sequence_path.iterdir()))
            image_path_list = image_path_list[(len(image_path_list)//2):]
            for image_path in image_path_list:
                image_id = image_path.stem 
                yield {
                    'sequence_id': sequence_id, 'frame_id': image_id
                }

    @staticmethod
    def write_split_config(cache_dir:Path, val_img_ids):
        out_file = cache_dir / 'splits' / 'centertrack_val.txt'
        out_file.parent.mkdir(parents=True, exist_ok=True)
        seq_dict = defaultdict(list)
        for item in val_img_ids:
            seq_dict[item['sequence_id']].append(int(item['frame_id']))
        with open(str(out_file), 'w') as f:
            for k in seq_dict.keys():
                min_, max_ = min(seq_dict[k]), max(seq_dict[k])
                f.write('{} empty {:06d} {:06d}\n'.format(k, min_, max_))


@SPLIT_REGISTRY.register
class AB3DMOTSplit:
    name='ab3dmot'
    val_set = (1, 6, 8, 10, 12, 13, 14, 15, 16, 18, 19)
    @classmethod
    def train_frameset_ids(cls, root:Union[Path, str]):
        if isinstance(root, str):
            root = Path(root)
        root_train = root / 'training'
        root_image_02 = root_train / 'image_02'
        for sequence_path in sorted(root_image_02.iterdir()):
            sequence_id = sequence_path.stem 
            if int(sequence_id) in cls.val_set: continue 
            image_path_list = sorted(list(sequence_path.iterdir()))
            for image_path in image_path_list:
                image_id = image_path.stem 
                yield {
                    'sequence_id': sequence_id, 'frame_id': image_id
                }

    @classmethod
    def val_frameset_ids(cls, root:Union[Path, str]):
        if isinstance(root, str):
            root = Path(root)
        root_train = root / 'training'
        root_image_02 = root_train / 'image_02'
        for sequence_path in sorted(root_image_02.iterdir()):
            sequence_id = sequence_path.stem 
            if int(sequence_id) not in cls.val_set: continue 
            image_path_list = sorted(list(sequence_path.iterdir()))
            for image_path in image_path_list:
                image_id = image_path.stem 
                yield {
                    'sequence_id': sequence_id, 'frame_id': image_id
                }          

    @staticmethod
    def test_frameset_ids(root:Union[Path, str]):
        if isinstance(root, str):
            root = Path(root)
        root_train = root / 'testing'
        root_image_02 = root_train / 'image_02'
        for sequence_path in sorted(root_image_02.iterdir()):
            sequence_id = sequence_path.stem 
            for image_path in sorted(sequence_path.iterdir()):
                image_id = image_path.stem 
                yield {
                    'sequence_id': sequence_id, 'frame_id': image_id
                }

    @staticmethod
    def write_split_config(cache_dir:Path, val_img_ids):
        out_file = cache_dir / 'splits' / 'ab3dmot_val.txt'
        out_file.parent.mkdir(parents=True, exist_ok=True)
        seq_dict = defaultdict(list)
        for item in val_img_ids:
            seq_dict[item['sequence_id']].append(int(item['frame_id']))
        with open(str(out_file), 'w') as f:
            for k in seq_dict.keys():
                min_, max_ = min(seq_dict[k]), max(seq_dict[k])
                f.write('{} empty {:06d} {:06d}\n'.format(k, min_, max_ + 1))

