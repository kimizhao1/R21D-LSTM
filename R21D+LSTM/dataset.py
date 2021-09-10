from typing import Tuple,List,Union
import numpy as np
from torch.utils.data import Dataset,DataLoader
from Sample import CreatSample

class MyDataset(Dataset):

    def __init__(self, clips: int, frames: int, size: Tuple[int,int], split: str='Train'):
        super(MyDataset, self).__init__()
        self.sampler = CreatSample(clips=clips,frames=frames,size=size,split=split)
        self.split = split
        if self.split == "Train":
            file_root = '.\\train.txt'
            num = 890
        elif self.split == "Test":
            file_root = '.\\val.txt'
            num = 227

        self.data = []
        with open(file_root,'r') as f:
            lines = f.readlines()[0:num]
            for each_line in lines:
                video_root,target = each_line.strip().split(' ')
                self.data.append({'video_root':video_root,'target':target})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        data = self.data[item]
        video_root,target = data['video_root'],data['target']
        clips = self.sampler(video_path=video_root)
        return clips,int(target)

if __name__ == '__main__':
    # dataset = MyDataset(3,8,(224,224))
    loader = DataLoader(dataset=MyDataset(clips=4,frames=32,size=(224,224)),batch_size=4,shuffle=True)

    for data in loader:
        inputs,targets = data
        for input in inputs:
            print(input.shape)
        print(targets)