import cv2
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# video_root = r'D:\Desktop\action_recognition\R3DCNN\UCF_data\UCF101\ApplyEyeMakeup\v_ApplyEyeMakeup_g11_c05.avi'

def clip_video(video_root: str,end_time: int):
    """
    将视频裁剪到固定的时长，若原视频总时长小于设置的end_time，则返回None
    :param video_root: 要处理视频的路径
    :param end_time: 要裁剪的时间长度
    """
    out_root = ".\\" + video_root.split('data\\')[1]
    dir = os.path.dirname(out_root)
    if not os.path.exists(dir):
        os.makedirs(dir)

    video = cv2.VideoCapture(video_root)
    assert video.isOpened(),"视频未正常打开"

    fps = video.get(cv2.CAP_PROP_FPS)
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    fourcc = int(video.get(cv2.CAP_PROP_FOURCC))
    frames = video.get(cv2.CAP_PROP_FRAME_COUNT)

    if frames/fps < end_time:
        return None
    count = 0
    out = cv2.VideoWriter(out_root,fourcc,fps,(width,height))
    while count < fps*end_time:
        ret,frame = video.read()
        assert ret,"视频帧无法正常打开"

        count += 1
        out.write(frame)

    video.release()
    out.release()

def batch_clips():
    root = r'D:\Desktop\action_recognition\R3DCNN\UCF_data\UCF101'
    action_names = os.listdir(root)
    for action_name in tqdm(action_names[20:101]):
        videos_root = os.path.join(root,action_name)
        video_names = os.listdir(videos_root)
        for video_name in video_names:
            video_root = os.path.join(videos_root,video_name)
            clip_video(video_root,end_time=5)

def Count():
    root = r'.\UCF101'
    Info = []
    action_names = os.listdir(root)
    for action_name in tqdm(action_names):
        videos_root = os.path.join(root, action_name)
        video_names = os.listdir(videos_root)
        Info.append((action_name,len(video_names)))
    return Info

def Ucf_Split(root):
    f1 = open('.\\train.txt','w')
    f2 = open('.\\val.txt','w')
    f3 = open('.\\Labels.txt','w')
    actions = os.listdir(root)
    for i,each in enumerate(actions):
        f3.write(each+'\n')
        videos_root = os.path.join(root,each)
        video_names = os.listdir(videos_root)
        root_list = [os.path.join(videos_root,v) for v in video_names]
        train,val = train_test_split(root_list,test_size=0.2,random_state=42)
        for each_root in train:
            f1.write(each_root+' '+str(i)+'\n')
        for each_root in val:
            f2.write(each_root+' '+str(i)+'\n')
    f1.close()
    f2.close()
    f3.close()

if __name__ == '__main__':
    # clip_video(video_root,5)
    # batch_clips()
    # Info = Count()
    # print(Info)
    Ucf_Split(root=r".\UCF101")

    pass
