import torch
import os
from models.RR21D import RR2_1D,trained_param
from torch.utils.data import DataLoader
from dataset import MyDataset
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

CUDA = True
Batch_size = 4
size = (112,112)
lr = 5*1e-5
clips = 4
frames = 16
num_classes = 10
frozen_epoch = 13
epoch = 30
frozen_param = True

class Train(object):
    def __init__(self):
        super(Train, self).__init__()
        self.model = RR2_1D(num_class=num_classes)
        self.softmax = torch.nn.Softmax(dim=1).cuda()

        self.criterion = torch.nn.CrossEntropyLoss()

        if frozen_param:
            check_point = torch.load('./result/model/RR2_1D-UCF-10_epoch-13_CrossELoss1_87.66519823788546.pth.tar',
                                    map_location='cpu')
            self.model.load_state_dict(check_point['state_dict'])
            param = trained_param(self.model)
            self.optimizer = torch.optim.Adam(param, lr=lr, weight_decay=5e-4)

        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=5e-4)

        self.schedule = torch.optim.lr_scheduler.StepLR(self.optimizer, 5, 0.95)

        self.Loder = {'train': DataLoader(MyDataset(clips=clips,frames=frames,size=size), batch_size=Batch_size, shuffle=True),
                 'Validation': DataLoader(MyDataset(clips=clips,frames=frames,size=size,split='Test'), batch_size=Batch_size, shuffle=True)}


        # if True:
        #     check_point = torch.load('./result/model/RR2_1D-UCF-10_epoch-5_CrossELoss.pth.tar',map_location='cpu')
        #     state_dict = check_point['state_dict']
        #     opt_dict = check_point['opt_dict']
        #
        #     self.model.load_state_dict(state_dict)
        #     self.optimizer.load_state_dict(opt_dict)
        #     for state in self.optimizer.state.values():
        #         for k, v in state.items():
        #             if torch.is_tensor(v):
        #                 state[k] = v.cuda(0)

        if CUDA:
            self.model.cuda()
            self.criterion.cuda()
            self.device = 'cuda:0'

    def train(self,epoch,writer,save_dir):
        # 一个epoch总误差和总正确识别的个数
        total_loss = 0.
        correct = 0

        each_correct = [0 for _ in range(clips)]

        # 模型设置为训练模式
        self.model.train()

        # 统计训练集中总共的数量
        total = 0
        epoch_size = self.Loder['train'].__len__()

        with tqdm(total=epoch_size,postfix=dict,mininterval=0.3,colour='blue') as bar:
            for iteration,data in enumerate(self.Loder['train']):
                inputs,labels = data
                batch_size = inputs[0].shape[0]

                inputs = [input.requires_grad_(True).float().to(self.device) for input in inputs]
                labels = labels.to(self.device)

                h = torch.zeros((batch_size,256)).to(self.device)
                c = torch.zeros((batch_size,256)).to(self.device)

                # 存储每一片段输出的结果
                outputs = torch.zeros((clips,batch_size,num_classes)).to(self.device)
                loss = torch.tensor(0.,dtype=torch.float32,device=self.device,requires_grad=True)

                for i,each_input in enumerate(inputs):
                    output,h,c = self.model(each_input,h,c)
                    outputs[i] = output

                loss = self.criterion(outputs[-1],labels.long())

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                for i,each_output in enumerate(outputs):
                    each_output = self.softmax(each_output)
                    pre = torch.max(each_output,dim=1)[1]
                    each_correct[i] += (pre == labels).sum().item()

                average = torch.mean(outputs,dim=0)
                predict = torch.max(average,dim=1)[1]
                correct += (predict == labels).sum().item()
                total_loss += loss.item()
                total += batch_size

                result = {'loss':format(total_loss/(iteration+1),".4f"),
                          'avrage_accuracy':correct/total*100,
                          }
                result.update({'accuracy_'+str(i+1):correct/total*100 for i,correct in enumerate(each_correct)})

                bar.set_postfix(result)

                bar.set_description('trian set {} epoch'.format(epoch+1))
                bar.update(1)

        self.schedule.step()

        # 每次epoch记录下训练误差和精度
        writer.add_scalar('epoch/train_loss',total_loss/epoch_size,epoch)
        writer.add_scalar('epoch/train_accuracy',correct/total,epoch)

        # 每隔5个epoch保存训练的模型参数
        if epoch % 5 == 4:
            dir = r".\result\model"
            if not os.path.exists(dir):
                os.makedirs(dir)
            torch.save({
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'opt_dict': self.optimizer.state_dict(),
            }, os.path.join(dir,'RR2_1D-UCF-10' + '_epoch-' + str(epoch+1) + '_CrossELoss' + '.pth.tar'))
            print('save model successfully!')

    def val(self,epoch,writer=None):
        correct = 0
        correct_last = 0
        total = 0
        self.model.eval()
        with torch.no_grad():
            for data in tqdm(self.Loder['Validation'],desc='验证集测试'):
                with torch.no_grad():
                    inputs, labels = data
                    batch_size = inputs[0].shape[0]

                    inputs = [input.float().to(self.device) for input in inputs]
                    labels = labels.to(self.device)

                    h = torch.zeros((batch_size, 256)).to(self.device)
                    c = torch.zeros((batch_size, 256)).to(self.device)

                    outputs = torch.zeros((clips,batch_size,num_classes)).to(self.device)

                    for i, each_input in enumerate(inputs):
                        output,h,c = self.model(each_input,h,c)
                        outputs[i] = output

                    average = torch.mean(outputs, dim=0)
                    prediction = torch.max(average,dim=1)[1]

                    prediction_last = torch.max(outputs[-1],dim=1)[1]

                    correct += (prediction == labels).sum().item()
                    correct_last += (prediction_last == labels).sum().item()
                    total += batch_size

        average_accuracy = 100*float(correct)/total
        accuracy_last = 100*float(correct_last)/total

        writer.add_scalar('epoch/val_average_accuracy',average_accuracy,epoch)
        writer.add_scalar('epoch/val_last_accuracy',accuracy_last,epoch)

        global val_max_accuracy
        if accuracy_last > val_max_accuracy:
            dir = r".\result\model"
            if not os.path.exists(dir):
                os.makedirs(dir)

            val_max_accuracy = accuracy_last
            torch.save({
                'state_dict': self.model.state_dict(),
            }, os.path.join(dir, 'RR2_1D-UCF-10' + '_epoch-' + str(epoch + 1) + '_CrossELoss_' + str(val_max_accuracy) +'.pth.tar'))

        print('Average accuracy on val set: {:.3f} % [{}/{}]'.format(100*correct/total,correct,total))
        print('Last accuracy on val set: {:.3f} % [{}/{}]'.format(100*correct_last/total,correct_last,total))

    def Begin(self):
        # 确定训练模型的可视化文件的位置
        save_root = '.\\result\\logs'
        if not os.path.exists(save_root):
            os.makedirs(save_root)
        writer = SummaryWriter(log_dir=save_root)

        for i in range(frozen_epoch,epoch):
            self.train(i,writer,save_root)
            self.val(i,writer)
        print('Finish the train!')
        writer.close()

if __name__ == '__main__':
    val_max_accuracy = 87.7
    train = Train()
    train.Begin()
