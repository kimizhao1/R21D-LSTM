import torch
import torchvision.models.video as video
from torch import nn

class RR2_1D(nn.Module):

    def __init__(self,num_class):
        super(RR2_1D, self).__init__()
        r2_1d = video.r2plus1d_18(pretrained=True)

        self.extract_feature = nn.Sequential(r2_1d.stem,
                                             r2_1d.layer1,
                                             r2_1d.layer2,
                                             r2_1d.layer3,
                                             r2_1d.layer4,
                                             r2_1d.avgpool)
        self.flatten = nn.Flatten()
        self.lstm = nn.LSTMCell(input_size=512,hidden_size=256)

        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(256,num_class)

        self.__init_weight()

    def forward(self,x,h,c):
        x = self.extract_feature(x)
        x = self.flatten(x)

        h,c = self.lstm(x,(h,c))
        h = self.dropout(h)

        output = self.fc(h)

        return output,h,c

    def __init_weight(self):
        layers = [self.lstm]
        for each in layers:
            for name, param in each.named_parameters():
                if 'weight' in name:
                    torch.nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    torch.nn.init.constant_(param, 0.)

        torch.nn.init.normal_(self.fc.weight, mean=0, std=0.01)
        torch.nn.init.zeros_(self.fc.bias)

def trained_param(model):
    """
    冻结RR21D模型中extractor的参数，仅训练lstm、fc层的参数
    :param model: 网络模型
    :return: 训练参数的iterator
    """
    layers = [model.lstm,model.fc]
    for i in range(len(layers)):
        for k in layers[i].parameters():
            if k.requires_grad:
                yield k
