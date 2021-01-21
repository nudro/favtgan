from __future__ import division, print_function, absolute_import
import re
import glob
import sys
import os
import os.path as osp
import warnings

from torchreid.data import ImageDataset



class Faces(ImageDataset): #class Faces


    dataset_dir = 'EAI_GANA_i'

    def __init__(self, root='', **kwargs): #(, **kwargs)
        self.root = osp.abspath(osp.expanduser(root)) # /reid-data
        self.dataset_dir = osp.join(self.root, self.dataset_dir) #/reid-data/omni

        self.train_dir = osp.join(self.dataset_dir, 'EAI_GANA_i', 'train') #reid-data/omni/omni-reid/train
        self.query_dir = osp.join(self.dataset_dir, 'EAI_GANA_i', 'probe') #reid-data/omni/omni-reid/probe
        self.gallery_dir = osp.join(self.dataset_dir, 'EAI_GANA_i', 'gallery') #reid-data/omni/omni-reid/gallery

        required_files = [
            self.dataset_dir, self.query_dir, self.gallery_dir, self.train_dir,
        ]

        self.check_before_run(required_files)

        train = self.process_dir(self.train_dir)
        query = self.process_dir(self.query_dir)
        gallery = self.process_dir(self.gallery_dir)

        super(Faces, self).__init__(train, query, gallery, **kwargs) #super(Faces)


    def process_dir(self, dir_path):
        img_paths = glob.glob(osp.join(dir_path, '*.png'))
        data = []

        for img_path in img_paths:
            #print("img_path:", img_path)
            img_name = osp.splitext(osp.basename(img_path))[0]
            #print("img_name:", img_name)
            pid, camid = img_name.split('_')
            pid, camid = int(pid), int(camid)
            data.append((img_path, pid, camid))

        return data
