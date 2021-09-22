# R21D-LSTM  
在R21D模型的基础上，添加了LSTM。该模型的训练使用的是UCF101数据集中的部分样本，由于采样器是以秒为单位，为了保证输入模型的长度都一样，所以
根据裁剪片段的数量（这里模型用的是clips=5），对UCF101数据集中小于5s的视频进行了剔除，然后为了保证每类动作都有足够的样本进行训练，对样本小
于100个的动作类别进行剔除，最终剩下59类动作，每类动作样本都大于100个。  
由于训练样本太少，验证集识别精度仅达到了88%，若继续训练，就会出现过拟合。  
其中predict.py进行动作识别，train.py进行模型训练，util.py包含数据集的筛选、裁剪和划分。

## Requirements:  
```bash
pytorch  
opencv  
torchvision  
tqdm  
numpy  
scikit-learn  
cudatoolkit  
```

## Pretrained Models:
仅提供前十个动作的训练权重(没有训练全部动作):  
[Baidu Netdisk](https://pan.baidu.com/s/1mf82d9keXXX4Zboq9VxjmA)----code：ah11  
