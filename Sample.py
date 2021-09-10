import cv2
import random
import time
import numpy as np
import torch.nn as nn
from typing import Tuple,List,Union

# class Sample(object):
#     """
#     针对于视频的以秒为单位的通用采样器
#     当用于模型训练时，仅具有采样器的作用；
#     当用于模型测试时，可以传入模型，进行预测
#     """
#     is_model = False
#
#     def __init__(self, *, clips: int, frames: int, size:Tuple[int,int], split: str = 'Train',model: nn.Module = None) -> None:
#         """
#         :param clips: 将视频分为若干个片段，一般该参数应结合视频秒数来决定，应该小于或等于视频总秒数
#         :param frames: 在片段中采样帧的数量，若该值大于视频FPS，则用一秒中的最后帧填充多出的部分
#         :param size: 采样图片的最终尺寸
#         :param split: 采样器的模式，默认为训练模式，optional：[Train,Test]
#         :param model: 若有训练好的模型，可以传入，进行预测
#         """
#         self.split = split
#         self.size = size
#         self.clips = clips
#         self.frames = frames
#         if model is not None and self.split == 'Test':
#             self.model = model
#             Sample.is_model = True
#
#     def __call__(self, *, video_path: str) -> List[np.ndarray,]:
#         """
#         返回一个列表，列表包含了每个片段的数据，每个片段又是一个numpy数组，包含了采样的帧
#         [[(3,frames,size[0],size[1])],[],...,[]]
#         :type video_path: 处理视频路径
#         """
#         try:
#             video = cv2.VideoCapture(video_path)
#         except Exception as e:
#             print('***视频未能正常打开***')
#             print(e)
#
#         fps = int(video.get(cv2.CAP_PROP_FPS))
#         video_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
#         video_seconds = int(video_frames/fps)
#
#         assert video_seconds>=self.clips,f"***视频秒数{video_seconds}应该大于片段数量{self.clips}，这里不满足条件***"
#         clips = self.Sample_clip_from_video(video=video,video_seconds=video_seconds,fps=fps)
#         return [self.transform(clip) for clip in clips]
#
#     def Sample_clip_from_video(self,video: cv2.VideoCapture,video_seconds: int,fps: int) -> List[List,]:
#         # 根据视频总长和片段数量，决定时间间隔
#         interval = int(video_seconds/self.clips)
#         # 准备存放片段的容器
#         clips = [[] for _ in range(self.clips)]
#         clip_count,frame_count = (0,0)
#
#         while clip_count<self.clips:
#             ret, frame = video.read()
#             assert ret,f"***视频流中存在无法打开的帧***"
#             # 进行图片帧的随机裁剪和归一操作
#             if self.split == 'Train':
#                 frame = self.random_crop(frame)
#             elif self.split == 'Test':
#                 frame = cv2.resize(frame, self.size, interpolation=cv2.INTER_CUBIC)
#
#             frame_count += 1
#             if frame_count<fps:
#                 clips[clip_count].append(frame)
#             elif frame_count==fps:
#                 # 如果视频每秒的帧数大于或等于每秒要采样的帧数，则采用随机采样的方式对视频帧采样
#                 if self.frames <= fps:
#                     clips[clip_count].append(frame)
#                     sample_index = sorted(
#                                           random.sample([each for each in range(fps)],k=self.frames),
#                                           reverse=False
#                                           )
#                     clips[clip_count] = [clips[clip_count][each] for each in sample_index]
#                 # 如果视频每秒的帧数小于每秒要采样的帧数,则用一秒中的最后帧填充多出的部分
#                 else:
#                     clips[clip_count].extend((frame for _ in range(self.frames - fps + 1)))
#
#                 # 若存在模型，可以用于预测每个clip的结果
#                 if Sample.is_model:
#                     clip = self.transform(clips[clip_count])
#                     pass
#
#                 clip_count+=1
#             else:
#                 if frame_count>interval*fps:
#                     clips[clip_count].append(frame)
#                     frame_count = 1
#         return clips
#
#     def random_crop(self,img):
#         height,width = img.shape[0:2]
#         random_height = height-self.size[0]
#         random_width = width-self.size[1]
#         try:
#             height_start = random.randint(0,random_height)
#             width_start = random.randint(0,random_width)
#             img = img[height_start:height_start+self.size[0],width_start:width_start+self.size[1],:]
#         except Exception as e:
#             print(f"***想要得到的尺寸{self.size}，超过原始图像的尺寸{img.shape}***")
#             print(e)
#         return img
#
#     def transform(self,clip:List[np.ndarray,]) -> np.ndarray:
#         # numpy数组转换
#         clip = np.array(clip,dtype=np.float32)
#         # 维度转换
#         clip = np.transpose(clip,(3,0,1,2))
#         # 归一化操作
#         clip = clip/255.
#         return clip

class CreatSample(object):
    """
    创建针对于视频的以秒为单位的通用采样器
    当用于模型训练时，仅具有采样器的作用；
    """

    def __init__(self, *, clips: int, frames: int, size:Tuple[int,int], split: str = 'Train') -> None:
        """
        :param clips: 将视频分为若干个片段，一般该参数应结合视频秒数来决定，应该小于或等于视频总秒数
        :param frames: 在片段中采样帧的数量，若该值大于视频FPS，则用一秒中的最后帧填充多出的部分
        :param size: 采样图片的最终尺寸
        :param split: 采样器的模式，默认为训练模式，optional：[Train,Test]
        """
        super(CreatSample, self).__init__()

        self.split = split
        self.size = size
        self.clips = clips
        self.frames = frames

    def __call__(self, *, video_path: str) -> List[np.ndarray,]:
        """
        返回一个列表，列表包含了每个片段的数据，每个片段又是一个numpy数组，包含了采样的帧
        [[(3,frames,size[0],size[1])],[],...,[]]
        :type video_path: 处理视频路径
        """
        try:
            video = cv2.VideoCapture(video_path)
        except Exception as e:
            print('***视频未能正常打开***')
            print(e)
        assert video.isOpened(),f"***视频未能正常打开***"

        fps = int(video.get(cv2.CAP_PROP_FPS))
        video_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
        video_seconds = int(video_frames/fps)

        assert video_seconds>=self.clips,f"***视频秒数{video_seconds}应该大于片段数量{self.clips}，这里不满足条件***"

        # 根据视频总长和片段数量，决定时间间隔
        interval = int(video_seconds / self.clips)
        start_time = 0.
        clips = []

        for _ in range(self.clips):
            clip,start_time = self.Sample_clip_from_video(video=video,interval=interval,fps=fps,start_time=start_time)
            clips.append(clip)

        return [self.transform(clip) for clip in clips]

    def Sample_clip_from_video(self,video: cv2.VideoCapture,interval: int,fps: int,start_time: float) -> Tuple[List[np.ndarray,],float]:
        """
        从视频中获得裁剪的片段
        :param video: 视频流
        :param interval: 有所需片段和视频总长决定，其值>=1
        :param fps: 帧速率
        :param start_time: 裁剪视频开始的时间，单位毫秒
        :return: 所裁剪的片段
        """
        # 准备存放片段的容器
        clip = []
        frame_count = 0
        video.set(cv2.CAP_PROP_POS_MSEC,value=start_time)
        while frame_count<(interval*fps+1):
            ret, frame = video.read()
            assert ret,f"***视频流中存在无法打开的帧***"

            frame_count += 1
            if frame_count<fps:
                clip.append(frame)
            elif frame_count==fps:
                # 如果视频每秒的帧数大于或等于每秒要采样的帧数，则采用随机采样的方式对视频帧采样
                if self.frames <= fps:
                    clip.append(frame)
                    sample_index = sorted(
                                          random.sample([each for each in range(fps)],k=self.frames),
                                          reverse=False
                                          )

                    # 进行图片帧的随机裁剪和归一操作
                    if self.split == "Train":
                        clip = [self.random_crop(clip[each]) for each in sample_index]
                    elif self.split == "Test":
                        clip = [self.center_clop(clip[each]) for each in sample_index]
                # 如果视频每秒的帧数小于每秒要采样的帧数,则用一秒中的最后帧填充多出的部分
                else:
                    clip.extend((frame for _ in range(self.frames - fps + 1)))
                    # 进行图片帧的随机裁剪和归一操作
                    if self.split == "Train":
                        clip = [self.random_crop(each) for each in clip]
                    elif self.split == "Test":
                        clip = [self.center_clop(each) for each in clip]

        end_time = video.get(0)
        return clip,end_time

    def random_crop(self,img: np.ndarray) -> np.ndarray:
        height,width = img.shape[0:2]
        random_height = height-self.size[0]
        random_width = width-self.size[1]
        try:
            height_start = random.randint(0,random_height)
            width_start = random.randint(0,random_width)
            img = img[height_start:height_start+self.size[0],width_start:width_start+self.size[1],:]
        except Exception as e:
            print(f"***想要得到的尺寸{self.size}，超过原始图像的尺寸{img.shape}***")
            print(e)

        return img

    def center_clop(self,img: np.ndarray) -> np.ndarray:
        height,width = img.shape[0:2]
        center_height = int((height-self.size[0])/2)
        center_width = int((width-self.size[1])/2)
        assert center_height>0 and center_width>0,f"***想要得到的尺寸{self.size}，超过原始图像的尺寸{img.shape}***"

        img = img[center_height:center_height+self.size[0],center_width:center_width+self.size[1],:]
        return img

    def scale(self,img: np.ndarray) -> np.ndarray:
        img = cv2.resize(img,self.size,interpolation=cv2.INTER_CUBIC)

        return img

    def transform(self,clip: List[np.ndarray,]) -> np.ndarray:
        # numpy数组转换
        clip = np.array(clip,dtype=np.float32)
        # 维度转换
        clip = np.transpose(clip,(3,0,1,2))
        # 归一化操作
        clip = clip/255.
        return clip



if __name__ == '__main__':
    video_root = r'.\UCF101\ApplyEyeMakeup\v_ApplyEyeMakeup_g07_c03.avi'
    sampler = CreatSample(clips=4,frames=32,size=(224,224),split='Test')
    clips = sampler(video_path=video_root)
    for clip in clips:
        print(clip.shape)