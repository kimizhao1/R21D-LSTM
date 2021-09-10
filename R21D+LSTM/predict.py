import torch
import cv2
import time
from Sample import CreatSample
from models.RR21D import RR2_1D

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
clips = 4
frames = 16

def predict(video_path: str,model: torch.nn.Module):
    check_point = torch.load('./result/model/RR2_1D-UCF-10_epoch-7_CrossELoss_88.10572687224669.pth.tar',map_location='cpu')
    model.load_state_dict(check_point['state_dict'])
    if device == 'cuda:0':
        model.cuda()
    model.eval()
    print('模型加载完毕！')

    with open('./Labels.txt','r') as f:
        labels = [label.strip() for label in f.readlines()]
    try:
        video = cv2.VideoCapture(video_path)
    except Exception as e:
        print('***视频未能正常打开***')
        print(e)
    assert video.isOpened(), f"***视频未能正常打开***"

    fps = int(video.get(cv2.CAP_PROP_FPS))
    video_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
    video_seconds = int(video_frames / fps)

    assert video_seconds >= clips, f"***视频秒数{video_seconds}应该大于片段数量{clips}，这里不满足条件***"

    # 根据视频总长和片段数量，决定时间间隔
    interval = int(video_seconds / clips)
    start_time = 0.

    sampler = CreatSample(clips=clips,frames=frames,size=(112,112),split='Test')
    softmax = torch.nn.Softmax(dim=1)

    h = torch.zeros((1, 256)).to(device)
    c = torch.zeros((1, 256)).to(device)
    for _ in range(clips):
        clip, start_time = sampler.Sample_clip_from_video(video=video, interval=interval, fps=fps, start_time=start_time)
        t0 = time.perf_counter()
        clip = sampler.transform(clip)
        input = torch.tensor([clip],dtype=torch.float32,device=device)
        output,h,c = model(input,h,c)
        used_time = time.perf_counter()-t0
        confi,pred = torch.max(softmax(output),dim=1)
        print("动作类别：{}，置信度：{}，识别时间：{:.2f}ms".format(labels[pred.item()],confi.item(),used_time*1000))

if __name__ == '__main__':
    # archery.mp4是训练集和验证集中都未出现的视频流，可以当测试样本使用
    video_path = r'.\archery.mp4'
    rr21d = RR2_1D(10)
    predict(video_path=video_path,model=rr21d)
