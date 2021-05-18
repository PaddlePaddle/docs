# **点云处理：实现PointNet点云分类**
**作者**：[Zhihao Cao](https://github.com/WhiteFireFox)<br>
**日期**：2021.05<br>
**摘要**：本示例在于演示如何基于Paddle2.1实现PointNet在ShapeNet数据集上进行点云分类处理。

## 一、环境设置

本教程基于Paddle 2.1 编写，如果你的环境不是本版本，请先参考官网[安装](https://www.paddlepaddle.org.cn/install/quick) Paddle 2.1 。


```python
import os
import numpy as np
import random
import h5py
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

print(paddle.__version__)
```

    2.1.0


## 二、数据集
### 2.1 数据介绍
ShapeNet数据集是一个注释丰富且规模较大的 3D 形状数据集，由斯坦福大学、普林斯顿大学和芝加哥丰田技术学院于 2015 年联合发布。<br>
ShapeNet数据集官方链接：[https://vision.princeton.edu/projects/2014/3DShapeNets/](https://vision.princeton.edu/projects/2014/3DShapeNets/)<br>
ShapeNet数据集的储存格式是h5文件，该文件中key值分别为：
- 1、data：这一份数据中所有点的xyz坐标，
- 2、label：这一份数据所属类别，如airplane等，
- 3、pid：这一份数据中所有点所属的类型，如这一份数据属airplane类，则它包含的所有点的类型有机翼、机身等类型。
### 2.2 解压数据集


```python
!unzip data/data70460/shapenet_part_seg_hdf5_data.zip
!mv hdf5_data dataset
```

    Archive:  data/data70460/shapenet_part_seg_hdf5_data.zip
       creating: hdf5_data/
      inflating: hdf5_data/ply_data_train5.h5  
      inflating: hdf5_data/ply_data_train1.h5  
      inflating: hdf5_data/ply_data_train3.h5  
      inflating: hdf5_data/ply_data_val0.h5  
      inflating: hdf5_data/ply_data_train0.h5  
      inflating: hdf5_data/ply_data_test1.h5  
      inflating: hdf5_data/ply_data_test0.h5  
      inflating: hdf5_data/ply_data_train4.h5  
      inflating: hdf5_data/ply_data_train2.h5  


### 2.3 数据列表
ShapeNet数据集所有的数据文件。


```python
train_list = ['ply_data_train0.h5', 'ply_data_train1.h5', 'ply_data_train2.h5', 'ply_data_train3.h5', 'ply_data_train4.h5', 'ply_data_train5.h5']
test_list = ['ply_data_test0.h5', 'ply_data_test1.h5']
val_list = ['ply_data_val0.h5']
```

### 2.4 搭建数据生成器
说明：将ShapeNet数据集全部读入。


```python
def make_data(mode='train', path='./dataset/', num_point=2048):
    datas = []
    labels = []
    if mode == 'train':
        for file_list in train_list:
            f = h5py.File(os.path.join(path, file_list), 'r')
            datas.extend(f['data'][:, :num_point, :])
            labels.extend(f['label'])
            f.close()
    elif mode == 'test':
        for file_list in test_list:
            f = h5py.File(os.path.join(path, file_list), 'r')
            datas.extend(f['data'][:, :num_point, :])
            labels.extend(f['label'])
            f.close()
    else:
        for file_list in val_list:
            f = h5py.File(os.path.join(path, file_list), 'r')
            datas.extend(f['data'][:, :num_point, :])
            labels.extend(f['label'])
            f.close()
    
    return datas, labels
```

说明：通过继承`paddle.io.Dataset`来完成数据集的构造。


```python
class PointDataset(paddle.io.Dataset):
    def __init__(self, datas, labels):
        super(PointDataset, self).__init__()
        self.datas = datas
        self.labels = labels
    
    def __getitem__(self, index):
        data = paddle.to_tensor(self.datas[index].T.astype('float32'))
        label = paddle.to_tensor(self.labels[index].astype('int64'))
        return data, label

    def __len__(self):
        return len(self.datas)
```

说明：使用飞桨框架提供的API：`paddle.io.DataLoader`完成数据的加载，使得按照Batchsize生成Mini-batch的数据。


```python
# 数据导入
datas, labels = make_data(mode='train', num_point=2048)
train_dataset = PointDataset(datas, labels)
datas, labels = make_data(mode='val', num_point=2048)
val_dataset = PointDataset(datas, labels)
datas, labels = make_data(mode='test', num_point=2048)
test_dataset = PointDataset(datas, labels)

# 实例化数据读取器
train_loader = paddle.io.DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=True,
    drop_last=False
)
val_loader = paddle.io.DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    drop_last=False
)
test_loader = paddle.io.DataLoader(
    test_dataset,
    batch_size=128,
    shuffle=False,
    drop_last=False
)
```

## 三、定义网络
PointNet是斯坦福大学研究人员提出的一个点云处理网络，在这篇论文中，它提出了空间变换网络（T-Net）解决点云的旋转问题（注：因为考虑到某一物体的点云旋转后还是该物体，所以需要有一个网络结构去学习并解决这个旋转问题），并且提出了采取MaxPooling的方法极大程度上地提取点云全局特征。

### 3.1 定义网络结构


```python
class PointNet(nn.Layer):
    def __init__(self, name_scope='PointNet_', num_classes=16, num_point=2048):
        super(PointNet, self).__init__()
        self.input_transform_net = nn.Sequential(
            nn.Conv1D(3, 64, 1),
            nn.BatchNorm(64),
            nn.ReLU(),
            nn.Conv1D(64, 128, 1),
            nn.BatchNorm(128),
            nn.ReLU(),
            nn.Conv1D(128, 1024, 1),
            nn.BatchNorm(1024),
            nn.ReLU(),
            nn.MaxPool1D(num_point)
        )
        self.input_fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 9,
                weight_attr=paddle.ParamAttr(initializer=paddle.nn.initializer.Assign(paddle.zeros((256, 9)))),
                bias_attr=paddle.ParamAttr(initializer=paddle.nn.initializer.Assign(paddle.reshape(paddle.eye(3), [-1])))
            )
        )
        self.mlp_1 = nn.Sequential(
            nn.Conv1D(3, 64, 1),
            nn.BatchNorm(64),
            nn.ReLU(),
            nn.Conv1D(64, 64, 1),
            nn.BatchNorm(64),
            nn.ReLU()
        )
        self.feature_transform_net = nn.Sequential(
            nn.Conv1D(64, 64, 1),
            nn.BatchNorm(64),
            nn.ReLU(),
            nn.Conv1D(64, 128, 1),
            nn.BatchNorm(128),
            nn.ReLU(),
            nn.Conv1D(128, 1024, 1),
            nn.BatchNorm(1024),
            nn.ReLU(),
            nn.MaxPool1D(num_point)
        )
        self.feature_fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 64*64)
        )
        self.mlp_2 = nn.Sequential(
            nn.Conv1D(64, 64, 1),
            nn.BatchNorm(64),
            nn.ReLU(),
            nn.Conv1D(64, 128, 1),
            nn.BatchNorm(128),
            nn.ReLU(),
            nn.Conv1D(128, 1024, 1),
            nn.BatchNorm(1024),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=0.7),
            nn.Linear(256, num_classes),
            nn.LogSoftmax(axis=-1)
        )
    def forward(self, inputs):
        batchsize = inputs.shape[0]

        t_net = self.input_transform_net(inputs)
        t_net = paddle.squeeze(t_net, axis=-1)
        t_net = self.input_fc(t_net)
        t_net = paddle.reshape(t_net, [batchsize, 3, 3])

        x = paddle.transpose(inputs, (0, 2, 1))
        x = paddle.matmul(x, t_net)
        x = paddle.transpose(x, (0, 2, 1))
        x = self.mlp_1(x)

        t_net = self.feature_transform_net(x)
        t_net = paddle.squeeze(t_net, axis=-1)
        t_net = self.feature_fc(t_net)
        t_net = paddle.reshape(t_net, [batchsize, 64, 64])

        x = paddle.squeeze(x, axis=-1)
        x = paddle.transpose(x, (0, 2, 1))
        x = paddle.matmul(x, t_net)
        x = paddle.transpose(x, (0, 2, 1))
        x = self.mlp_2(x)
        x = paddle.max(x, axis=-1)
        x = paddle.squeeze(x, axis=-1)
        x = self.fc(x)

        return x
```

### 3.2 网络结构可视化
说明：使用飞桨API：`paddle.summary`完成模型结构可视化


```python
pointnet = PointNet()
paddle.summary(pointnet, (64, 3, 2048))
```

    ---------------------------------------------------------------------------
     Layer (type)       Input Shape          Output Shape         Param #    
    ===========================================================================
       Conv1D-1       [[64, 3, 2048]]       [64, 64, 2048]          256      
      BatchNorm-1     [[64, 64, 2048]]      [64, 64, 2048]          256      
        ReLU-1        [[64, 64, 2048]]      [64, 64, 2048]           0       
       Conv1D-2       [[64, 64, 2048]]     [64, 128, 2048]         8,320     
      BatchNorm-2    [[64, 128, 2048]]     [64, 128, 2048]          512      
        ReLU-2       [[64, 128, 2048]]     [64, 128, 2048]           0       
       Conv1D-3      [[64, 128, 2048]]     [64, 1024, 2048]       132,096    
      BatchNorm-3    [[64, 1024, 2048]]    [64, 1024, 2048]        4,096     
        ReLU-3       [[64, 1024, 2048]]    [64, 1024, 2048]          0       
      MaxPool1D-1    [[64, 1024, 2048]]     [64, 1024, 1]            0       
       Linear-1         [[64, 1024]]          [64, 512]           524,800    
        ReLU-4          [[64, 512]]           [64, 512]              0       
       Linear-2         [[64, 512]]           [64, 256]           131,328    
        ReLU-5          [[64, 256]]           [64, 256]              0       
       Linear-3         [[64, 256]]            [64, 9]             2,313     
       Conv1D-4       [[64, 3, 2048]]       [64, 64, 2048]          256      
      BatchNorm-4     [[64, 64, 2048]]      [64, 64, 2048]          256      
        ReLU-6        [[64, 64, 2048]]      [64, 64, 2048]           0       
       Conv1D-5       [[64, 64, 2048]]      [64, 64, 2048]         4,160     
      BatchNorm-5     [[64, 64, 2048]]      [64, 64, 2048]          256      
        ReLU-7        [[64, 64, 2048]]      [64, 64, 2048]           0       
       Conv1D-6       [[64, 64, 2048]]      [64, 64, 2048]         4,160     
      BatchNorm-6     [[64, 64, 2048]]      [64, 64, 2048]          256      
        ReLU-8        [[64, 64, 2048]]      [64, 64, 2048]           0       
       Conv1D-7       [[64, 64, 2048]]     [64, 128, 2048]         8,320     
      BatchNorm-7    [[64, 128, 2048]]     [64, 128, 2048]          512      
        ReLU-9       [[64, 128, 2048]]     [64, 128, 2048]           0       
       Conv1D-8      [[64, 128, 2048]]     [64, 1024, 2048]       132,096    
      BatchNorm-8    [[64, 1024, 2048]]    [64, 1024, 2048]        4,096     
        ReLU-10      [[64, 1024, 2048]]    [64, 1024, 2048]          0       
      MaxPool1D-2    [[64, 1024, 2048]]     [64, 1024, 1]            0       
       Linear-4         [[64, 1024]]          [64, 512]           524,800    
        ReLU-11         [[64, 512]]           [64, 512]              0       
       Linear-5         [[64, 512]]           [64, 256]           131,328    
        ReLU-12         [[64, 256]]           [64, 256]              0       
       Linear-6         [[64, 256]]           [64, 4096]         1,052,672   
       Conv1D-9       [[64, 64, 2048]]      [64, 64, 2048]         4,160     
      BatchNorm-9     [[64, 64, 2048]]      [64, 64, 2048]          256      
        ReLU-13       [[64, 64, 2048]]      [64, 64, 2048]           0       
       Conv1D-10      [[64, 64, 2048]]     [64, 128, 2048]         8,320     
     BatchNorm-10    [[64, 128, 2048]]     [64, 128, 2048]          512      
        ReLU-14      [[64, 128, 2048]]     [64, 128, 2048]           0       
       Conv1D-11     [[64, 128, 2048]]     [64, 1024, 2048]       132,096    
     BatchNorm-11    [[64, 1024, 2048]]    [64, 1024, 2048]        4,096     
        ReLU-15      [[64, 1024, 2048]]    [64, 1024, 2048]          0       
       Linear-7         [[64, 1024]]          [64, 512]           524,800    
        ReLU-16         [[64, 512]]           [64, 512]              0       
       Linear-8         [[64, 512]]           [64, 256]           131,328    
        ReLU-17         [[64, 256]]           [64, 256]              0       
       Dropout-1        [[64, 256]]           [64, 256]              0       
       Linear-9         [[64, 256]]            [64, 16]            4,112     
     LogSoftmax-1        [[64, 16]]            [64, 16]              0       
    ===========================================================================
    Total params: 3,476,825
    Trainable params: 3,461,721
    Non-trainable params: 15,104
    ---------------------------------------------------------------------------
    Input size (MB): 1.50
    Forward/backward pass size (MB): 11333.40
    Params size (MB): 13.26
    Estimated Total Size (MB): 11348.16
    ---------------------------------------------------------------------------
    





    {'total_params': 3476825, 'trainable_params': 3461721}



## 四、训练
说明：模型训练的时候，将会使用`paddle.optimizer.Adam`优化器来进行优化。使用`F.nll_loss`来计算损失值。


```python
def train():
    model = PointNet(num_classes=16, num_point=2048)
    model.train()
    optim = paddle.optimizer.Adam(parameters=model.parameters(), weight_decay=0.001)

    epoch_num = 10
    for epoch in range(epoch_num):
        # train
        print("===================================train===========================================")
        for batch_id, data in enumerate(train_loader()):
            inputs, labels = data

            predicts = model(inputs)
            loss = F.nll_loss(predicts, labels)
            acc = paddle.metric.accuracy(predicts, labels)        

            if batch_id % 20 == 0: 
                print("train: epoch: {}, batch_id: {}, loss is: {}, accuracy is: {}".format(epoch, batch_id, loss.numpy(), acc.numpy()))
            
            loss.backward()
            optim.step()
            optim.clear_grad()

        if epoch % 2 == 0:
            paddle.save(model.state_dict(), './model/PointNet.pdparams')
            paddle.save(optim.state_dict(), './model/PointNet.pdopt')
        
        # validation
        print("===================================val===========================================")
        model.eval()
        accuracies = []
        losses = []
        for batch_id, data in enumerate(val_loader()):
            inputs, labels = data

            predicts = model(inputs)

            loss = F.nll_loss(predicts, labels)
            acc = paddle.metric.accuracy(predicts, labels)    
            
            losses.append(loss.numpy())
            accuracies.append(acc.numpy())

        avg_acc, avg_loss = np.mean(accuracies), np.mean(losses)
        print("validation: loss is: {}, accuracy is: {}".format(avg_loss, avg_acc))
        model.train()

if __name__ == '__main__':
    train()
```

    ===================================train===========================================
    train: epoch: 0, batch_id: 0, loss is: [4.9248095], accuracy is: [0.1484375]
    train: epoch: 0, batch_id: 20, loss is: [1.2662703], accuracy is: [0.6875]
    train: epoch: 0, batch_id: 40, loss is: [0.79613143], accuracy is: [0.7734375]
    train: epoch: 0, batch_id: 60, loss is: [0.5991034], accuracy is: [0.8203125]
    train: epoch: 0, batch_id: 80, loss is: [0.6732792], accuracy is: [0.8125]
    ===================================val===========================================
    validation: loss is: 0.41937708854675293, accuracy is: 0.8569915294647217
    ===================================train===========================================
    train: epoch: 1, batch_id: 0, loss is: [0.5992371], accuracy is: [0.8046875]
    train: epoch: 1, batch_id: 20, loss is: [0.65309006], accuracy is: [0.84375]
    train: epoch: 1, batch_id: 40, loss is: [0.4504382], accuracy is: [0.8671875]
    train: epoch: 1, batch_id: 60, loss is: [0.5059966], accuracy is: [0.828125]
    train: epoch: 1, batch_id: 80, loss is: [0.28158492], accuracy is: [0.875]
    ===================================val===========================================
    validation: loss is: 0.30018773674964905, accuracy is: 0.90625
    ===================================train===========================================
    train: epoch: 2, batch_id: 0, loss is: [0.27240375], accuracy is: [0.9375]
    train: epoch: 2, batch_id: 20, loss is: [0.4211054], accuracy is: [0.9140625]
    train: epoch: 2, batch_id: 40, loss is: [0.3876957], accuracy is: [0.890625]
    train: epoch: 2, batch_id: 60, loss is: [0.27216607], accuracy is: [0.9140625]
    train: epoch: 2, batch_id: 80, loss is: [0.29216224], accuracy is: [0.9296875]
    ===================================val===========================================
    validation: loss is: 6.236376762390137, accuracy is: 0.07642251998186111
    ===================================train===========================================
    train: epoch: 3, batch_id: 0, loss is: [2.7821736], accuracy is: [0.1015625]
    train: epoch: 3, batch_id: 20, loss is: [2.5189795], accuracy is: [0.2109375]
    train: epoch: 3, batch_id: 40, loss is: [2.2503586], accuracy is: [0.2109375]
    train: epoch: 3, batch_id: 60, loss is: [2.1081736], accuracy is: [0.328125]
    train: epoch: 3, batch_id: 80, loss is: [1.9944972], accuracy is: [0.375]
    ===================================val===========================================
    validation: loss is: 1505.8250732421875, accuracy is: 0.2111077606678009
    ===================================train===========================================
    train: epoch: 4, batch_id: 0, loss is: [2.041934], accuracy is: [0.3203125]
    train: epoch: 4, batch_id: 20, loss is: [2.013423], accuracy is: [0.3203125]
    train: epoch: 4, batch_id: 40, loss is: [2.1235168], accuracy is: [0.328125]
    train: epoch: 4, batch_id: 60, loss is: [2.0412865], accuracy is: [0.2578125]
    train: epoch: 4, batch_id: 80, loss is: [2.0814402], accuracy is: [0.328125]
    ===================================val===========================================
    validation: loss is: 2.312401533126831, accuracy is: 0.3141646683216095
    ===================================train===========================================
    train: epoch: 5, batch_id: 0, loss is: [2.343786], accuracy is: [0.21875]
    train: epoch: 5, batch_id: 20, loss is: [2.121391], accuracy is: [0.296875]
    train: epoch: 5, batch_id: 40, loss is: [1.9093541], accuracy is: [0.3828125]
    train: epoch: 5, batch_id: 60, loss is: [2.181718], accuracy is: [0.296875]
    train: epoch: 5, batch_id: 80, loss is: [1.9160612], accuracy is: [0.3359375]
    ===================================val===========================================
    validation: loss is: 2.033637046813965, accuracy is: 0.3141646683216095
    ===================================train===========================================
    train: epoch: 6, batch_id: 0, loss is: [2.0708382], accuracy is: [0.3203125]
    train: epoch: 6, batch_id: 20, loss is: [1.8967582], accuracy is: [0.3515625]
    train: epoch: 6, batch_id: 40, loss is: [1.9126657], accuracy is: [0.390625]
    train: epoch: 6, batch_id: 60, loss is: [1.9337373], accuracy is: [0.3125]
    train: epoch: 6, batch_id: 80, loss is: [1.9214094], accuracy is: [0.34375]
    ===================================val===========================================
    validation: loss is: 2.130458116531372, accuracy is: 0.3141646683216095
    ===================================train===========================================
    train: epoch: 7, batch_id: 0, loss is: [2.105646], accuracy is: [0.2734375]
    train: epoch: 7, batch_id: 20, loss is: [2.063758], accuracy is: [0.28125]
    train: epoch: 7, batch_id: 40, loss is: [2.0195878], accuracy is: [0.328125]
    train: epoch: 7, batch_id: 60, loss is: [2.061315], accuracy is: [0.265625]
    train: epoch: 7, batch_id: 80, loss is: [2.0420928], accuracy is: [0.2578125]
    ===================================val===========================================
    validation: loss is: 2.0498859882354736, accuracy is: 0.3141646683216095
    ===================================train===========================================
    train: epoch: 8, batch_id: 0, loss is: [1.9741396], accuracy is: [0.3515625]
    train: epoch: 8, batch_id: 20, loss is: [2.0922964], accuracy is: [0.28125]
    train: epoch: 8, batch_id: 40, loss is: [1.926232], accuracy is: [0.3125]
    train: epoch: 8, batch_id: 60, loss is: [2.044017], accuracy is: [0.3828125]
    train: epoch: 8, batch_id: 80, loss is: [2.0386844], accuracy is: [0.296875]
    ===================================val===========================================
    validation: loss is: 1.9772093296051025, accuracy is: 0.3141646683216095
    ===================================train===========================================
    train: epoch: 9, batch_id: 0, loss is: [2.0979326], accuracy is: [0.3125]
    train: epoch: 9, batch_id: 20, loss is: [2.0070634], accuracy is: [0.2734375]
    train: epoch: 9, batch_id: 40, loss is: [2.0131662], accuracy is: [0.3046875]
    train: epoch: 9, batch_id: 60, loss is: [2.030023], accuracy is: [0.3046875]
    train: epoch: 9, batch_id: 80, loss is: [2.210213], accuracy is: [0.2890625]
    ===================================val===========================================
    validation: loss is: 2.011220693588257, accuracy is: 0.3141646683216095


## 五、评估与测试
说明：通过`model.load_dict`的方式加载训练好的模型对测试集上的数据进行评估与测试。


```python
def evaluation():
    model = PointNet()
    model_state_dict = paddle.load('./model/PointNet.pdparams')
    model.load_dict(model_state_dict)

    model.eval()
    accuracies = []
    losses = []
    for batch_id, data in enumerate(test_loader()):
        inputs, labels = data

        predicts = model(inputs)

        loss = F.nll_loss(predicts, labels)
        acc = paddle.metric.accuracy(predicts, labels)    
        
        losses.append(loss.numpy())
        accuracies.append(acc.numpy())

    avg_acc, avg_loss = np.mean(accuracies), np.mean(losses)
    print("validation: loss is: {}, accuracy is: {}".format(avg_loss, avg_acc))

if __name__ == '__main__':
    evaluation()
```
