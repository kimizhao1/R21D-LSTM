# R21D-LSTM  
在R21D模型的基础上，添加了LSTM，来看看效果怎么样。  
需要安装的API:  
pytorch  
opencv  
torchvision  
tqdm  
numpy  
scikit-learn  
cudatoolkit(如果没有GPU可以不装)  
该模型的训练使用的是UCF101数据集中的部分样本，由于采样器是以秒为单位，为了保证输入模型的长度都一样，所以
根据裁剪片段的数量（这里模型用的是clips=5），对UCF101数据集中小于5s的视频进行了剔除，然后为了保证每类动
作都有足够的样本进行训练，对样本小于100个的动作类别进行剔除，最终剩下59类动作，每类动作样本都大于100个。  
其中predict.py进行动作识别，train.py进行模型训练，util.py包含数据集的筛选、裁剪和划分。  
