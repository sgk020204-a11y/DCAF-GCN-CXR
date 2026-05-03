import torch.utils.data as data
import json
import os
import subprocess
from PIL import Image
import numpy as np
import torch
import pickle
from util import *

urls = {'train_img':'http://images.cocodataset.org/zips/train2014.zip',
        'val_img' : 'http://images.cocodataset.org/zips/val2014.zip',
        'annotations':'http://images.cocodataset.org/annotations/annotations_trainval2014.zip'}


class COCO2014(data.Dataset):
    def __init__(self, root, transform=None, phase='train', inp_name=None):
        """COCO2014 数据集初始化
        Args:
            root: 数据集根目录路径
            transform: 图像预处理变换
            phase: 数据阶段 (train/val)
            inp_name: 预训练词向量文件路径
        """
        self.root = root
        self.phase = phase
        self.img_list = []  # 存储图像信息列表
        self.transform = transform
        self.get_anno()  # 加载标注信息
        self.num_classes = len(self.cat2idx)  # 获取类别总数

        # 加载预训练的词向量
        with open(inp_name, 'rb') as f:
            self.inp = pickle.load(f)
        self.inp_name = inp_name  # 保存词向量路径

    def get_anno(self):
        """加载标注文件和类别映射表"""
        list_path = os.path.join(self.root, '{}_anno.json'.format(self.phase))
        self.img_list = json.load(open(list_path, 'r'))  # 加载图像标注列表
        self.cat2idx = json.load(open(os.path.join(self.root, 'category.json'), 'r'))  # 类别名称到索引的映射

    def __len__(self):
        """返回数据集的总样本数"""
        return len(self.img_list)

    def __getitem__(self, index):
        """获取单个样本数据"""
        item = self.img_list[index]
        return self.get(item)

    def get(self, item):
        """处理单个样本的具体加载逻辑
        Returns:
            tuple: (图像张量, 文件名, 词向量), 多标签目标向量
        """
        filename = item['file_name']
        labels = sorted(item['labels'])  # 排序后的标签索引列表
        
        # 加载并转换图像
        img = Image.open(os.path.join(self.root, '{}2014'.format(self.phase), filename)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)  # 应用图像变换
            
        # 创建多标签目标向量（-1表示负样本，1表示正样本）
        target = np.full(self.num_classes, -1, dtype=np.float32)
        target[labels] = 1  # 设置存在标签的位置为1
        
        return (img, filename, self.inp), target







import pandas as pd
import numpy as np
import pickle as pk
import copy
import gc
from torch.utils.data import Dataset, DataLoader
from os.path import join
from PIL import Image
from utility.preprocessing import adj_from_series
from utility.selfdefine import FlexCounter
from heapq import nlargest
from collections import Counter


from torch.utils.data import Dataset
from torchvision import transforms
import torch
from PIL import Image
import pandas as pd
from os.path import join

class Chexpert_Dataset(Dataset):
    """Chexpert dataset."""
    def __init__(self, path='/share/fsmresfiles/CheXpert-v1.0-small/', mode='RGB', adjgroup=True, neib_samp='relation', 
                 relations=['pid', 'age', 'gender', 'view'], k=16, graph_nodes='current', transform=None, split='random', inp_name=None):
        self.path = path
        csv_labelfile = join(path, 'train_val.csv')
        self.all_label_df = pd.read_csv(csv_labelfile)
        self.all_label_df.fillna(0, inplace=True)
        self.label_df = self.all_label_df
        self.mode = mode
        self.adjgroup = adjgroup
        self.neib_samp = neib_samp
        self.k = k
        self.gnode = graph_nodes
        self.transform = transform
        self.relations = relations
        self.all_grp = self.creat_adj(self.label_df)
        self.grp = self.all_grp
        self.num_classes = 14  # 获取类别总数

        # 数据集分割
        self.tr_set, self.val_set, self.te_set = self.tr_val_te_split(split=split)

                # 加载预训练的词向量
        with open(inp_name, 'rb') as f:
            self.inp = pickle.load(f)
        self.inp_name = inp_name  # 保存词向量路径

    def __len__(self):
        return len(self.label_df)

    def __getitem__(self, idx):
        # 获取图像数据和标签
        sample = self._getimage(idx)
        if self.neib_samp == 'relation':
            img = self.label_df.iloc[idx]
            impt = self.impt_sample(img, k=1) 
            sample['impt'] = impt

        # 将图像转换为张量
        if self.transform is not None:
            sample['image'] = self.transform(sample['image'])

        # 生成多标签目标向量
        target = np.zeros(self.num_classes, np.float32) - 1
        target[sample['label'] == 1] = 1

        # 返回元组，包含图像、文件名、词向量和目标向量
        return (sample['image'], sample['name'], self.inp), target

    def _getimage(self, idx, byindex=False, level=0):
        img = self.all_label_df.loc[idx] if byindex else self.label_df.iloc[idx]
        image = Image.open(join(self.path, img['Path'])).convert(self.mode)
        
        # 提取标签（Chexpert数据集标签从第6列到第18列）
        labels = img[6:20].astype(float).values
        
        sample = {'image': image, 'label': labels, 'pid': img['pid'], 'age': img['age'], 'gender': img['gender'], 
                  'view': img['view'], 'name': img['Path'], 'index': img.name}
        
        return sample

    
    def impt_sample(self, img, method='relation', k=1, base='train'):
        """
        为img采样重要的k个样本。
        方法：“样本”——按概率随机选择
        “最好的”——选择最重要的
        “关系”——每个关系随机选择k
        基础：选择基础集，“训练”或“全部”
            返回：重要样本的索引列表
        """
        if base == "train":
            grps = self.tr_grp
        elif base == "all":
            grps = self.all_grp 
            
        if method=='relation':
            impt_sample=[]
            for r, grp in grps.items():
                if img[r] in grp:
                    neibs = grp[img[r]].drop(img.name, errors = 'ignore')
                    if not neibs.empty:
                        impt_sample += np.random.choice(neibs, k, replace=False).tolist()
            return impt_sample    
    
        w = sum([FlexCounter(grp[img[r]])/len(grp[img[r]]) for r, grp in grps.items()], Counter())
        w.pop(img.name, None)
        if method == "sample":   
            p = FlexCounter(w)/sum(w.values())
            impt_sample = np.random.choice(list(p.keys()), k, replace=False, p=list(p.values()))
        elif method == 'best':
            impt_sample = nlargest(k, w, key = w.get) 
            
        return impt_sample
    
    def creat_adj(self, label_df, adjgroup=True ):
        if self.gnode=='current':
            adj = {r:adj_from_series(label_df[r], groups= adjgroup) for r in self.relations}
        else:
            pass
        return adj
            
    def tr_val_te_split(self,  split='random', tr_pct=0.7 ):
        n_all=len(self.all_label_df) 
        np.random.seed(0)
        if split=='specified':
            perm = np.random.permutation(223414)
            tr_df = self.all_label_df.iloc[perm[:int(223414*0.9)]]
            val_df = self.all_label_df.iloc[perm[int(223414*0.9):]]
            te_df = self.all_label_df.iloc[-234:]
        elif split == 'random':
            tr_df, val_df, te_df = np.split(self.all_label_df.sample(frac=1, random_state=0),[int(n_all*0.7), int(n_all*0.8),])
            tr_df = tr_df.sample(n=int(n_all*tr_pct),random_state=0)
        else:
            raise Error('split %s is not defined'%(split))
        
        self.tr_grp = self.creat_adj(tr_df)
        
        tr_set, val_set, te_set = copy.copy(self), copy.copy(self), copy.copy(self)
        tr_set.label_df, val_set.label_df, te_set.label_df  = tr_df, val_df, te_df

        
        return tr_set, val_set, te_set


class MimicCXR_Dataset(Dataset):
    """MimicCXR dataset."""
    
    def __init__(self, path='/share/fsmresfiles/MIMIC_CXR/v1.0/MIMICCXR/', mode='RGB', adjgroup=True, neib_samp='relation', \
                 relations= ['pid', 'view'], k = 16, graph_nodes='current',  transform=None,split='random', inp_name=None):
        self.path=path
        csv_labelfile=join(path, 'train_val.csv')
        # csv_bboxfile=join(path,'BBox_list_2017.csv')
        self.all_label_df = pd.read_csv(csv_labelfile)
        self.all_label_df.fillna(0,inplace=True)
        self.label_df = self.all_label_df
        self.mode = mode
        self.adjgroup = adjgroup
        self.neib_samp = neib_samp
        self.k=k
        self.gnode = graph_nodes

        self.transform = transform
        self.relations = relations
        self.all_grp = self.creat_adj(self.label_df ) 
        self.grp = self.all_grp

                # 数据集分割
        self.tr_set, self.val_set, self.te_set = self.tr_val_te_split(split=split)

                # 加载预训练的词向量
        with open(inp_name, 'rb') as f:
            self.inp = pickle.load(f)
        self.inp_name = inp_name  # 保存词向量路径

    def __len__(self):
        return len(self.label_df)

    def __getitem__(self, idx):
        # 获取图像数据和标签
        sample = self._getimage(idx)
        if self.neib_samp == 'relation':
            img = self.label_df.iloc[idx]
            impt = self.impt_sample(img, k=1) 
            sample['impt'] = impt

        # 将图像转换为张量
        if self.transform is not None:
            sample['image'] = self.transform(sample['image'])

        # 生成多标签目标向量
        target = np.zeros(self.num_classes, np.float32) - 1
        target[sample['label'] == 1] = 1

        # 返回元组，包含图像、文件名、词向量和目标向量
        return (sample['image'], sample['name'], self.inp), target
    
    def _getimage(self, idx, byindex=False, level=0):
        img = self.all_label_df.loc[idx] if byindex else self.label_df.iloc[idx]
        image = Image.open(join(self.path, img['path'])).convert(self.mode)        
        
        labels= img[3:16].astype(float).values
        sample = {'image': image, 'label': labels, 'pid':img['pid'],  \
                  'view':img['view'],  'name': img['path'], 'index':img.name}
        if level==0:
            sample['dataset'] = self
            if self.neib_samp in ('sampling', 'best'):
                w = sum([(FlexCounter(grp[img[r]])/len(grp[img[r]]) if img[r] in  grp else FlexCounter())   \
                         for r, grp in self.tr_grp.items()], Counter())
                sample['weight'] = w           

        if self.transform is not None:
            sample['image'] = self.transform(sample['image'])

        return sample
    
    def impt_sample(self, img, method='relation', k=1, base='train'):
        """
        sampling the important k samples for img.
        method: "sample"--random choose by probability
                "best"--choose the most important
                "relation"--random choose k for each relation
        base: choose the basic set, "train" or "all"
        """
        if base == "train":
            grps = self.tr_grp
        elif base == "all":
            grps = self.all_grp 
            
        if method=='relation':
            impt_sample=[]
            for r, grp in grps.items():
                if img[r]!='other' and img[r] in grp:
                    neibs = grp[img[r]].drop(img.name, errors = 'ignore')
                    if not neibs.empty:
                        impt_sample += np.random.choice(neibs, k, replace=False).tolist()
            return impt_sample    
    
        w = sum([FlexCounter(grp[img[r]])/len(grp[img[r]]) for r, grp in grps.items()], Counter())
        w.pop(img.name, None)
        if method == "sample":   
            p = FlexCounter(w)/sum(w.values())
            impt_sample = np.random.choice(list(p.keys()), k, replace=False, p=list(p.values()))
        elif method == 'best':
            impt_sample = nlargest(k, w, key = w.get) 
            
        return impt_sample
    
    def creat_adj(self, label_df, adjgroup=True ):
        if self.gnode=='current':
            adj = {r:adj_from_series(label_df[r], groups= adjgroup) for r in self.relations}
        else:
            pass
        return adj
            
    def tr_val_te_split(self,  split='random', tr_pct=0.7 ):
        n_all=len(self.all_label_df) 
        np.random.seed(0)
        if split=='specified':
            perm = np.random.permutation(369188)
            tr_df = self.all_label_df.iloc[perm[:int(369188*0.9)]]
            val_df = self.all_label_df.iloc[perm[int(369188*0.9):]]
            te_df = self.all_label_df.iloc[-2732:]
        elif split == 'random':
            tr_df, val_df, te_df = np.split(self.all_label_df.sample(frac=1, random_state=0),[int(n_all*0.7), int(n_all*0.8),])
            tr_df = tr_df.sample(n=int(n_all*tr_pct),random_state=0)
        else:
            raise Error('split %s is not defined'%(split))
        
        self.tr_grp = self.creat_adj(tr_df)
        
        tr_set, val_set, te_set = copy.copy(self), copy.copy(self), copy.copy(self)
        tr_set.label_df, val_set.label_df, te_set.label_df  = tr_df, val_df, te_df

        
        return tr_set, val_set, te_set
