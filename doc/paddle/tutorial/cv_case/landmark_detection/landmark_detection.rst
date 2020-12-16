关键点检测方法及应用1: 人脸关键点检测
=====================================

本示例教程当前是基于2.0-rc1版本Paddle做的案例实现，未来会随着2.0的系列版本发布进行升级。

1.简要介绍
----------

在图像处理中，关键点本质上是一种特征。它是对一个固定区域或者空间物理关系的抽象描述，描述的是一定邻域范围内的组合或上下文关系。它不仅仅是一个点信息，或代表一个位置，更代表着上下文与周围邻域的组合关系。关键点检测的目标就是通过计算机从图像中找出这些点的坐标，作为计算机视觉领域的一个基础任务，关键点的检测对于高级别任务，例如识别和分类具有至关重要的意义。

关键点检测方法总体上可以分成两个类型，一个种是用坐标回归的方式来解决，另一种是将关键点建模成热力图，通过像素分类任务，回归热力图分布得到关键点位置。这两个方法，都是一种手段或者是途径，解决的问题就是要找出这个点在图像当中的位置与关系。

其中人脸关键点检测是关键点检测方法的一个成功实践，本示例简要介绍如何通过飞桨开源框架，实现人脸关键点检测的功能。这个案例用到的是第一种关键点检测方法——坐标回归。我们将使用到新版的paddle2.0的API，集成式的训练接口，能够很方便对模型进行训练和预测。

2.环境设置
----------

导入必要的模块，确认自己的飞桨版本。
如果是cpu环境，请安装cpu版本的paddle2.0环境，在 ``paddle.set_device()``
输入对应运行设备。

.. code:: ipython3

    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    import os
    
    import paddle
    from paddle.io import Dataset
    from paddle.vision.transforms import transforms
    from paddle.vision.models import resnet18
    from paddle.nn import functional as F
    print(paddle.__version__)
    device = paddle.set_device('cpu') 
    #device = paddle.set_device('gpu') 


.. parsed-literal::

    2.0.0-rc1


3.数据集
--------

3.1 数据集下载
~~~~~~~~~~~~~~

本案例使用了Kaggle官方举办的人脸关键点检测challenge数据集，官网：\ https://www.kaggle.com/c/facial-keypoints-detection

官方数据集将人脸图像和标注数据打包成了csv文件，我们使用panda来读取。其中数据集中的文件：
training.csv: 包含了用于训练的人脸关键点坐标和图像。 test.csv:
包含了用于测试的人脸关键点图像, 没有标注关键点坐标。 IdLookupTable.csv:
测试集关键点的位置的对应名称。

图像的长和宽都为96像素，所需要检测的一共有15个关键点。

.. code:: ipython3

    !unzip -o data/data60/test.zip -d data/data60/
    !unzip -o data/data60/training.zip -d data/data60


.. parsed-literal::

    Archive:  data/data60/test.zip
      inflating: data/data60/test.csv    
    Archive:  data/data60/training.zip
      inflating: data/data60/training.csv  


3.2 数据集定义
~~~~~~~~~~~~~~

飞桨（PaddlePaddle）数据集加载方案是统一使用Dataset（数据集定义） +
DataLoader（多进程数据集加载）。

首先我们先进行数据集的定义，数据集定义主要是实现一个新的Dataset类，继承父类paddle.io.Dataset，并实现父类中以下两个抽象方法，__getitem__和\ **len**\ ：

.. code:: ipython3

    Train_Dir = 'data/data60//training.csv'
    Test_Dir = 'data/data60//test.csv'
    lookid_dir = 'data/data60/IdLookupTable.csv'
    class ImgTransforms(object):
        """
        图像预处理工具，用于将图像进行升维(96, 96) => (96, 96, 3)，
        并对图像的维度进行转换从HWC变为CHW
        """
        def __init__(self, fmt):
            self.format = fmt
    
        def __call__(self, img):
            if len(img.shape) == 2:
                img = np.expand_dims(img, axis=2)
            img =  img.transpose(self.format)
    
            if img.shape[0] == 1:
                img = np.repeat(img, 3, axis=0)
            return img
    
    class FaceDataset(Dataset):
        def __init__(self, data_path, mode='train', val_split=0.2):
            self.mode = mode
            assert self.mode in ['train', 'val', 'test'], \
                "mode should be 'train' or 'test', but got {}".format(self.mode)
            self.data_source = pd.read_csv(data_path)
            # 清洗数据, 数据集中有很多样本只标注了部分关键点, 这里有两种策略
            # 第一种, 将未标注的位置从上一个样本对应的关键点复制过来
            # self.data_source.fillna(method = 'ffill',inplace = True)
            # 第二种, 将包含有未标注的样本从数据集中移除
            self.data_source.dropna(how="any", inplace=True)  
            self.data_label_all = self.data_source.drop('Image', axis = 1)
            
            # 划分训练集和验证集合
            if self.mode in ['train', 'val']:
                np.random.seed(43)
                data_len = len(self.data_source)
                # 随机划分
                shuffled_indices = np.random.permutation(data_len)
                # 顺序划分
                # shuffled_indices = np.arange(data_len)
                self.shuffled_indices = shuffled_indices
                val_set_size = int(data_len*val_split)
                if self.mode == 'val':
                    val_indices = shuffled_indices[:val_set_size]
                    self.data_img = self.data_source.reindex().iloc[val_indices]
                    self.data_label = self.data_label_all.reindex().iloc[val_indices]
                elif self.mode == 'train':
                    train_indices = shuffled_indices[val_set_size:]
                    self.data_img = self.data_source.reindex().iloc[train_indices]
                    self.data_label = self.data_label_all.reindex().iloc[train_indices]
            elif self.mode == 'test':
                self.data_img = self.data_source
                self.data_label = self.data_label_all
    
            self.transforms = transforms.Compose([
                ImgTransforms((2, 0, 1))
            ])
    
        # 每次迭代时返回数据和对应的标签
        def __getitem__(self, idx):
    
            img = self.data_img['Image'].iloc[idx].split(' ')
            img = ['0' if x == '' else x for x in img]
            img = np.array(img, dtype = 'float32').reshape(96, 96)
            img = self.transforms(img)
            label = np.array(self.data_label.iloc[idx,:],dtype = 'float32')/96
            return img, label
    
        # 返回整个数据集的总数
        def __len__(self):
            return len(self.data_img)
    # 训练数据集和验证数据集
    train_dataset = FaceDataset(Train_Dir, mode='train')
    val_dataset = FaceDataset(Train_Dir, mode='val')
    
    # 测试数据集
    test_dataset = FaceDataset(Test_Dir,  mode='test')

3.3 数据集抽样展示
~~~~~~~~~~~~~~~~~~

实现好Dataset数据集后，我们来测试一下数据集是否符合预期，因为Dataset是一个可以被迭代的Class，我们通过for循环从里面读取数据来用matplotlib进行展示。关键点的坐标在数据集中进行了归一化处理，这里乘以图像的大小恢复到原始尺度，并用scatter函数将点画在输出的图像上。

.. code:: ipython3

    def plot_sample(x, y, axis):
        img = x.reshape(96, 96)
        axis.imshow(img, cmap='gray')
        axis.scatter(y[0::2], y[1::2], marker='x', s=10, color='b')
    
    fig = plt.figure(figsize=(10, 7))
    fig.subplots_adjust(
        left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
    
    # 随机取16个样本展示
    for i in range(16):
        axis = fig.add_subplot(4, 4, i+1, xticks=[], yticks=[])
        idx = np.random.randint(train_dataset.__len__())
        # print(idx)
        img, label = train_dataset[idx]
        label = label*96
        plot_sample(img[0], label, axis)
    plt.show()



.. image:: landmark_detection_files/landmark_detection_8_0.png


4.定义模型
----------

这里使用到paddle.vision.models中定义的resnet18网络模型。在ImageNet分类任务中，图像分成1000类，在模型后接一个全连接层，将输出的1000维向量映射成30维，对应15个关键点的横纵坐标。

.. code:: ipython3

    class FaceNet(paddle.nn.Layer):
        def __init__(self, num_keypoints, pretrained=False):
            super(FaceNet, self).__init__()
            self.backbone = resnet18(pretrained)
            self.outLayer1 = paddle.nn.Sequential(
                paddle.nn.Linear(1000, 512),
                paddle.nn.ReLU(),
                paddle.nn.Dropout(0.1))
            self.outLayer2 = paddle.nn.Linear(512, num_keypoints*2)
        def forward(self, inputs):
            out = self.backbone(inputs)
            out = self.outLayer1(out)
            out = self.outLayer2(out)
            return out

4.1 模型可视化
~~~~~~~~~~~~~~

调用飞桨提供的summary接口对组建好的模型进行可视化，方便进行模型结构和参数信息的查看和确认。

.. code:: ipython3

    from paddle.static import InputSpec
    
    num_keypoints = 15
    model = paddle.Model(FaceNet(num_keypoints))
    model.summary((1,3, 96, 96))


.. parsed-literal::

    -------------------------------------------------------------------------------
       Layer (type)         Input Shape          Output Shape         Param #    
    ===============================================================================
         Conv2D-21        [[1, 3, 96, 96]]     [1, 64, 48, 48]         9,408     
      BatchNorm2D-21     [[1, 64, 48, 48]]     [1, 64, 48, 48]          256      
          ReLU-11        [[1, 64, 48, 48]]     [1, 64, 48, 48]           0       
        MaxPool2D-2      [[1, 64, 48, 48]]     [1, 64, 24, 24]           0       
         Conv2D-22       [[1, 64, 24, 24]]     [1, 64, 24, 24]        36,864     
      BatchNorm2D-22     [[1, 64, 24, 24]]     [1, 64, 24, 24]          256      
          ReLU-12        [[1, 64, 24, 24]]     [1, 64, 24, 24]           0       
         Conv2D-23       [[1, 64, 24, 24]]     [1, 64, 24, 24]        36,864     
      BatchNorm2D-23     [[1, 64, 24, 24]]     [1, 64, 24, 24]          256      
       BasicBlock-9      [[1, 64, 24, 24]]     [1, 64, 24, 24]           0       
         Conv2D-24       [[1, 64, 24, 24]]     [1, 64, 24, 24]        36,864     
      BatchNorm2D-24     [[1, 64, 24, 24]]     [1, 64, 24, 24]          256      
          ReLU-13        [[1, 64, 24, 24]]     [1, 64, 24, 24]           0       
         Conv2D-25       [[1, 64, 24, 24]]     [1, 64, 24, 24]        36,864     
      BatchNorm2D-25     [[1, 64, 24, 24]]     [1, 64, 24, 24]          256      
       BasicBlock-10     [[1, 64, 24, 24]]     [1, 64, 24, 24]           0       
         Conv2D-27       [[1, 64, 24, 24]]     [1, 128, 12, 12]       73,728     
      BatchNorm2D-27     [[1, 128, 12, 12]]    [1, 128, 12, 12]         512      
          ReLU-14        [[1, 128, 12, 12]]    [1, 128, 12, 12]          0       
         Conv2D-28       [[1, 128, 12, 12]]    [1, 128, 12, 12]       147,456    
      BatchNorm2D-28     [[1, 128, 12, 12]]    [1, 128, 12, 12]         512      
         Conv2D-26       [[1, 64, 24, 24]]     [1, 128, 12, 12]        8,192     
      BatchNorm2D-26     [[1, 128, 12, 12]]    [1, 128, 12, 12]         512      
       BasicBlock-11     [[1, 64, 24, 24]]     [1, 128, 12, 12]          0       
         Conv2D-29       [[1, 128, 12, 12]]    [1, 128, 12, 12]       147,456    
      BatchNorm2D-29     [[1, 128, 12, 12]]    [1, 128, 12, 12]         512      
          ReLU-15        [[1, 128, 12, 12]]    [1, 128, 12, 12]          0       
         Conv2D-30       [[1, 128, 12, 12]]    [1, 128, 12, 12]       147,456    
      BatchNorm2D-30     [[1, 128, 12, 12]]    [1, 128, 12, 12]         512      
       BasicBlock-12     [[1, 128, 12, 12]]    [1, 128, 12, 12]          0       
         Conv2D-32       [[1, 128, 12, 12]]     [1, 256, 6, 6]        294,912    
      BatchNorm2D-32      [[1, 256, 6, 6]]      [1, 256, 6, 6]         1,024     
          ReLU-16         [[1, 256, 6, 6]]      [1, 256, 6, 6]           0       
         Conv2D-33        [[1, 256, 6, 6]]      [1, 256, 6, 6]        589,824    
      BatchNorm2D-33      [[1, 256, 6, 6]]      [1, 256, 6, 6]         1,024     
         Conv2D-31       [[1, 128, 12, 12]]     [1, 256, 6, 6]        32,768     
      BatchNorm2D-31      [[1, 256, 6, 6]]      [1, 256, 6, 6]         1,024     
       BasicBlock-13     [[1, 128, 12, 12]]     [1, 256, 6, 6]           0       
         Conv2D-34        [[1, 256, 6, 6]]      [1, 256, 6, 6]        589,824    
      BatchNorm2D-34      [[1, 256, 6, 6]]      [1, 256, 6, 6]         1,024     
          ReLU-17         [[1, 256, 6, 6]]      [1, 256, 6, 6]           0       
         Conv2D-35        [[1, 256, 6, 6]]      [1, 256, 6, 6]        589,824    
      BatchNorm2D-35      [[1, 256, 6, 6]]      [1, 256, 6, 6]         1,024     
       BasicBlock-14      [[1, 256, 6, 6]]      [1, 256, 6, 6]           0       
         Conv2D-37        [[1, 256, 6, 6]]      [1, 512, 3, 3]       1,179,648   
      BatchNorm2D-37      [[1, 512, 3, 3]]      [1, 512, 3, 3]         2,048     
          ReLU-18         [[1, 512, 3, 3]]      [1, 512, 3, 3]           0       
         Conv2D-38        [[1, 512, 3, 3]]      [1, 512, 3, 3]       2,359,296   
      BatchNorm2D-38      [[1, 512, 3, 3]]      [1, 512, 3, 3]         2,048     
         Conv2D-36        [[1, 256, 6, 6]]      [1, 512, 3, 3]        131,072    
      BatchNorm2D-36      [[1, 512, 3, 3]]      [1, 512, 3, 3]         2,048     
       BasicBlock-15      [[1, 256, 6, 6]]      [1, 512, 3, 3]           0       
         Conv2D-39        [[1, 512, 3, 3]]      [1, 512, 3, 3]       2,359,296   
      BatchNorm2D-39      [[1, 512, 3, 3]]      [1, 512, 3, 3]         2,048     
          ReLU-19         [[1, 512, 3, 3]]      [1, 512, 3, 3]           0       
         Conv2D-40        [[1, 512, 3, 3]]      [1, 512, 3, 3]       2,359,296   
      BatchNorm2D-40      [[1, 512, 3, 3]]      [1, 512, 3, 3]         2,048     
       BasicBlock-16      [[1, 512, 3, 3]]      [1, 512, 3, 3]           0       
    AdaptiveAvgPool2D-2   [[1, 512, 3, 3]]      [1, 512, 1, 1]           0       
         Linear-4            [[1, 512]]           [1, 1000]           513,000    
         ResNet-2         [[1, 3, 96, 96]]        [1, 1000]              0       
         Linear-5           [[1, 1000]]            [1, 512]           512,512    
          ReLU-20            [[1, 512]]            [1, 512]              0       
         Dropout-2           [[1, 512]]            [1, 512]              0       
         Linear-6            [[1, 512]]            [1, 30]            15,390     
    ===============================================================================
    Total params: 12,227,014
    Trainable params: 12,207,814
    Non-trainable params: 19,200
    -------------------------------------------------------------------------------
    Input size (MB): 0.11
    Forward/backward pass size (MB): 10.51
    Params size (MB): 46.64
    Estimated Total Size (MB): 57.26
    -------------------------------------------------------------------------------
    




.. parsed-literal::

    {'total_params': 12227014, 'trainable_params': 12207814}



5. 训练模型
-----------

在这个任务是对坐标进行回归，我们使用均方误差（Mean Square error
）损失函数\ ``paddle.nn.MSELoss()``\ 来做计算，飞桨2.0中，在nn下将损失函数封装成可调用类。我们这里使用paddle.Model相关的API直接进行训练，只需要定义好数据集、网络模型和损失函数即可。

5.1 启动模型训练
----------------

使用模型代码进行Model实例生成，使用prepare接口定义优化器、损失函数和评价指标等信息，用于后续训练使用。在所有初步配置完成后，调用fit接口开启训练执行过程，调用fit时只需要将前面定义好的训练数据集、测试数据集、训练轮次（Epoch）和批次大小（batch_size）配置好即可。

.. code:: ipython3

    model = paddle.Model(FaceNet(num_keypoints=15))
    optim = paddle.optimizer.Adam(learning_rate=1e-3,
        parameters=model.parameters())
    model.prepare(optim, paddle.nn.MSELoss())
    model.fit(train_dataset, val_dataset, epochs=60, batch_size=256)


.. parsed-literal::

    The loss value printed in the log is the current step, and the metric is the average value of previous step.
    Epoch 1/60
    step 7/7 - loss: 0.0938 - 11s/step
    Eval begin...
    The loss value printed in the log is the current batch, and the metric is the average value of previous step.
    step 2/2 - loss: 20.6454 - 4s/step
    Eval samples: 428
    Epoch 2/60
    step 7/7 - loss: 0.0362 - 11s/step
    Eval begin...
    The loss value printed in the log is the current batch, and the metric is the average value of previous step.
    step 2/2 - loss: 1.0127 - 4s/step
    Eval samples: 428
    Epoch 3/60
    step 7/7 - loss: 0.0243 - 12s/step
    Eval begin...
    The loss value printed in the log is the current batch, and the metric is the average value of previous step.
    step 2/2 - loss: 0.1299 - 4s/step
    Eval samples: 428
    Epoch 4/60
    step 7/7 - loss: 0.0181 - 11s/step
    Eval begin...
    The loss value printed in the log is the current batch, and the metric is the average value of previous step.
    step 2/2 - loss: 0.0318 - 4s/step
    Eval samples: 428
    Epoch 5/60
    step 7/7 - loss: 0.0143 - 12s/step
    Eval begin...
    The loss value printed in the log is the current batch, and the metric is the average value of previous step.
    step 2/2 - loss: 0.0055 - 4s/step
    Eval samples: 428
    Epoch 6/60
    step 7/7 - loss: 0.0128 - 12s/step
    Eval begin...
    The loss value printed in the log is the current batch, and the metric is the average value of previous step.
    step 2/2 - loss: 0.0027 - 4s/step
    Eval samples: 428
    Epoch 7/60
    step 7/7 - loss: 0.0119 - 12s/step
    Eval begin...
    The loss value printed in the log is the current batch, and the metric is the average value of previous step.
    step 2/2 - loss: 0.0022 - 4s/step
    Eval samples: 428
    Epoch 8/60
    step 7/7 - loss: 0.0102 - 12s/step
    Eval begin...
    The loss value printed in the log is the current batch, and the metric is the average value of previous step.
    step 2/2 - loss: 0.0017 - 4s/step
    Eval samples: 428
    Epoch 9/60
    step 7/7 - loss: 0.0093 - 12s/step
    Eval begin...
    The loss value printed in the log is the current batch, and the metric is the average value of previous step.
    step 2/2 - loss: 0.0019 - 4s/step
    Eval samples: 428
    Epoch 10/60
    step 7/7 - loss: 0.0088 - 11s/step
    Eval begin...
    The loss value printed in the log is the current batch, and the metric is the average value of previous step.
    step 2/2 - loss: 0.0030 - 4s/step
    Eval samples: 428
    Epoch 11/60
    step 7/7 - loss: 0.0090 - 12s/step
    Eval begin...
    The loss value printed in the log is the current batch, and the metric is the average value of previous step.
    step 2/2 - loss: 0.0013 - 4s/step
    Eval samples: 428
    Epoch 12/60
    step 7/7 - loss: 0.0107 - 12s/step
    Eval begin...
    The loss value printed in the log is the current batch, and the metric is the average value of previous step.
    step 2/2 - loss: 0.0022 - 4s/step
    Eval samples: 428
    Epoch 13/60
    step 7/7 - loss: 0.0105 - 11s/step
    Eval begin...
    The loss value printed in the log is the current batch, and the metric is the average value of previous step.
    step 2/2 - loss: 0.0077 - 4s/step
    Eval samples: 428
    Epoch 14/60
    step 7/7 - loss: 0.0074 - 11s/step
    Eval begin...
    The loss value printed in the log is the current batch, and the metric is the average value of previous step.
    step 2/2 - loss: 0.0055 - 4s/step
    Eval samples: 428
    Epoch 15/60
    step 7/7 - loss: 0.0092 - 11s/step
    Eval begin...
    The loss value printed in the log is the current batch, and the metric is the average value of previous step.
    step 2/2 - loss: 0.0045 - 4s/step
    Eval samples: 428
    Epoch 16/60
    step 7/7 - loss: 0.0090 - 12s/step
    Eval begin...
    The loss value printed in the log is the current batch, and the metric is the average value of previous step.
    step 2/2 - loss: 0.0040 - 4s/step
    Eval samples: 428
    Epoch 17/60
    step 7/7 - loss: 0.0085 - 11s/step
    Eval begin...
    The loss value printed in the log is the current batch, and the metric is the average value of previous step.
    step 2/2 - loss: 0.0011 - 4s/step
    Eval samples: 428
    Epoch 18/60
    step 7/7 - loss: 0.0070 - 11s/step
    Eval begin...
    The loss value printed in the log is the current batch, and the metric is the average value of previous step.
    step 2/2 - loss: 9.3820e-04 - 3s/step
    Eval samples: 428
    Epoch 19/60
    step 7/7 - loss: 0.0060 - 11s/step
    Eval begin...
    The loss value printed in the log is the current batch, and the metric is the average value of previous step.
    step 2/2 - loss: 0.0013 - 4s/step
    Eval samples: 428
    Epoch 20/60
    step 7/7 - loss: 0.0063 - 11s/step
    Eval begin...
    The loss value printed in the log is the current batch, and the metric is the average value of previous step.
    step 2/2 - loss: 0.0012 - 3s/step
    Eval samples: 428
    Epoch 21/60
    step 7/7 - loss: 0.0067 - 11s/step
    Eval begin...
    The loss value printed in the log is the current batch, and the metric is the average value of previous step.
    step 2/2 - loss: 9.9358e-04 - 4s/step
    Eval samples: 428
    Epoch 22/60
    step 7/7 - loss: 0.0063 - 11s/step
    Eval begin...
    The loss value printed in the log is the current batch, and the metric is the average value of previous step.
    step 2/2 - loss: 0.0018 - 4s/step
    Eval samples: 428
    Epoch 23/60
    step 7/7 - loss: 0.0090 - 11s/step
    Eval begin...
    The loss value printed in the log is the current batch, and the metric is the average value of previous step.
    step 2/2 - loss: 0.0053 - 4s/step
    Eval samples: 428
    Epoch 24/60
    step 7/7 - loss: 0.0074 - 11s/step
    Eval begin...
    The loss value printed in the log is the current batch, and the metric is the average value of previous step.
    step 2/2 - loss: 0.0021 - 4s/step
    Eval samples: 428
    Epoch 25/60
    step 7/7 - loss: 0.0081 - 11s/step
    Eval begin...
    The loss value printed in the log is the current batch, and the metric is the average value of previous step.
    step 2/2 - loss: 0.0048 - 4s/step
    Eval samples: 428
    Epoch 26/60
    step 7/7 - loss: 0.0100 - 11s/step
    Eval begin...
    The loss value printed in the log is the current batch, and the metric is the average value of previous step.
    step 2/2 - loss: 0.0025 - 4s/step
    Eval samples: 428
    Epoch 27/60
    step 7/7 - loss: 0.0050 - 11s/step
    Eval begin...
    The loss value printed in the log is the current batch, and the metric is the average value of previous step.
    step 2/2 - loss: 0.0012 - 4s/step
    Eval samples: 428
    Epoch 28/60
    step 7/7 - loss: 0.0051 - 11s/step
    Eval begin...
    The loss value printed in the log is the current batch, and the metric is the average value of previous step.
    step 2/2 - loss: 0.0015 - 4s/step
    Eval samples: 428
    Epoch 29/60
    step 7/7 - loss: 0.0050 - 11s/step
    Eval begin...
    The loss value printed in the log is the current batch, and the metric is the average value of previous step.
    step 2/2 - loss: 0.0017 - 4s/step
    Eval samples: 428
    Epoch 30/60
    step 7/7 - loss: 0.0055 - 11s/step
    Eval begin...
    The loss value printed in the log is the current batch, and the metric is the average value of previous step.
    step 2/2 - loss: 0.0010 - 4s/step
    Eval samples: 428
    Epoch 31/60
    step 7/7 - loss: 0.0128 - 11s/step
    Eval begin...
    The loss value printed in the log is the current batch, and the metric is the average value of previous step.
    step 2/2 - loss: 0.0083 - 4s/step
    Eval samples: 428
    Epoch 32/60
    step 7/7 - loss: 0.0070 - 12s/step
    Eval begin...
    The loss value printed in the log is the current batch, and the metric is the average value of previous step.
    step 2/2 - loss: 0.0011 - 4s/step
    Eval samples: 428
    Epoch 33/60
    step 7/7 - loss: 0.0056 - 11s/step
    Eval begin...
    The loss value printed in the log is the current batch, and the metric is the average value of previous step.
    step 2/2 - loss: 0.0019 - 4s/step
    Eval samples: 428
    Epoch 34/60
    step 7/7 - loss: 0.0068 - 13s/step
    Eval begin...
    The loss value printed in the log is the current batch, and the metric is the average value of previous step.
    step 2/2 - loss: 0.0028 - 4s/step
    Eval samples: 428
    Epoch 35/60
    step 7/7 - loss: 0.0057 - 12s/step
    Eval begin...
    The loss value printed in the log is the current batch, and the metric is the average value of previous step.
    step 2/2 - loss: 0.0037 - 4s/step
    Eval samples: 428
    Epoch 36/60
    step 7/7 - loss: 0.0049 - 13s/step
    Eval begin...
    The loss value printed in the log is the current batch, and the metric is the average value of previous step.
    step 2/2 - loss: 0.0010 - 4s/step
    Eval samples: 428
    Epoch 37/60
    step 7/7 - loss: 0.0045 - 13s/step
    Eval begin...
    The loss value printed in the log is the current batch, and the metric is the average value of previous step.
    step 2/2 - loss: 0.0014 - 4s/step
    Eval samples: 428
    Epoch 38/60
    step 7/7 - loss: 0.0042 - 13s/step
    Eval begin...
    The loss value printed in the log is the current batch, and the metric is the average value of previous step.
    step 2/2 - loss: 7.7749e-04 - 4s/step
    Eval samples: 428
    Epoch 39/60
    step 7/7 - loss: 0.0047 - 13s/step
    Eval begin...
    The loss value printed in the log is the current batch, and the metric is the average value of previous step.
    step 2/2 - loss: 0.0015 - 4s/step
    Eval samples: 428
    Epoch 40/60
    step 7/7 - loss: 0.0047 - 12s/step
    Eval begin...
    The loss value printed in the log is the current batch, and the metric is the average value of previous step.
    step 2/2 - loss: 0.0010 - 4s/step
    Eval samples: 428
    Epoch 41/60
    step 7/7 - loss: 0.0038 - 13s/step
    Eval begin...
    The loss value printed in the log is the current batch, and the metric is the average value of previous step.
    step 2/2 - loss: 7.4004e-04 - 4s/step
    Eval samples: 428
    Epoch 42/60
    step 7/7 - loss: 0.0043 - 12s/step
    Eval begin...
    The loss value printed in the log is the current batch, and the metric is the average value of previous step.
    step 2/2 - loss: 0.0022 - 4s/step
    Eval samples: 428
    Epoch 43/60
    step 7/7 - loss: 0.0052 - 13s/step
    Eval begin...
    The loss value printed in the log is the current batch, and the metric is the average value of previous step.
    step 2/2 - loss: 0.0013 - 4s/step
    Eval samples: 428
    Epoch 44/60
    step 7/7 - loss: 0.0037 - 13s/step
    Eval begin...
    The loss value printed in the log is the current batch, and the metric is the average value of previous step.
    step 2/2 - loss: 8.0787e-04 - 4s/step
    Eval samples: 428
    Epoch 45/60
    step 7/7 - loss: 0.0043 - 13s/step
    Eval begin...
    The loss value printed in the log is the current batch, and the metric is the average value of previous step.
    step 2/2 - loss: 7.1024e-04 - 4s/step
    Eval samples: 428
    Epoch 46/60
    step 7/7 - loss: 0.0040 - 13s/step
    Eval begin...
    The loss value printed in the log is the current batch, and the metric is the average value of previous step.
    step 2/2 - loss: 0.0013 - 4s/step
    Eval samples: 428
    Epoch 47/60
    step 7/7 - loss: 0.0042 - 12s/step
    Eval begin...
    The loss value printed in the log is the current batch, and the metric is the average value of previous step.
    step 2/2 - loss: 0.0021 - 4s/step
    Eval samples: 428
    Epoch 48/60
    step 7/7 - loss: 0.0056 - 13s/step
    Eval begin...
    The loss value printed in the log is the current batch, and the metric is the average value of previous step.
    step 2/2 - loss: 0.0036 - 4s/step
    Eval samples: 428
    Epoch 49/60
    step 7/7 - loss: 0.0043 - 12s/step
    Eval begin...
    The loss value printed in the log is the current batch, and the metric is the average value of previous step.
    step 2/2 - loss: 0.0018 - 4s/step
    Eval samples: 428
    Epoch 50/60
    step 7/7 - loss: 0.0059 - 12s/step
    Eval begin...
    The loss value printed in the log is the current batch, and the metric is the average value of previous step.
    step 2/2 - loss: 0.0023 - 4s/step
    Eval samples: 428
    Epoch 51/60
    step 7/7 - loss: 0.0039 - 12s/step
    Eval begin...
    The loss value printed in the log is the current batch, and the metric is the average value of previous step.
    step 2/2 - loss: 7.2327e-04 - 4s/step
    Eval samples: 428
    Epoch 52/60
    step 7/7 - loss: 0.0042 - 12s/step
    Eval begin...
    The loss value printed in the log is the current batch, and the metric is the average value of previous step.
    step 2/2 - loss: 0.0022 - 4s/step
    Eval samples: 428
    Epoch 53/60
    step 7/7 - loss: 0.0034 - 12s/step
    Eval begin...
    The loss value printed in the log is the current batch, and the metric is the average value of previous step.
    step 2/2 - loss: 7.6005e-04 - 4s/step
    Eval samples: 428
    Epoch 54/60
    step 7/7 - loss: 0.0038 - 12s/step
    Eval begin...
    The loss value printed in the log is the current batch, and the metric is the average value of previous step.
    step 2/2 - loss: 0.0011 - 4s/step
    Eval samples: 428
    Epoch 55/60
    step 7/7 - loss: 0.0038 - 11s/step
    Eval begin...
    The loss value printed in the log is the current batch, and the metric is the average value of previous step.
    step 2/2 - loss: 7.1396e-04 - 4s/step
    Eval samples: 428
    Epoch 56/60
    step 7/7 - loss: 0.0034 - 12s/step
    Eval begin...
    The loss value printed in the log is the current batch, and the metric is the average value of previous step.
    step 2/2 - loss: 0.0032 - 4s/step
    Eval samples: 428
    Epoch 57/60
    step 7/7 - loss: 0.0039 - 12s/step
    Eval begin...
    The loss value printed in the log is the current batch, and the metric is the average value of previous step.
    step 2/2 - loss: 0.0013 - 4s/step
    Eval samples: 428
    Epoch 58/60
    step 7/7 - loss: 0.0033 - 12s/step
    Eval begin...
    The loss value printed in the log is the current batch, and the metric is the average value of previous step.
    step 2/2 - loss: 0.0017 - 4s/step
    Eval samples: 428
    Epoch 59/60
    step 7/7 - loss: 0.0037 - 12s/step
    Eval begin...
    The loss value printed in the log is the current batch, and the metric is the average value of previous step.
    step 2/2 - loss: 6.9502e-04 - 4s/step
    Eval samples: 428
    Epoch 60/60
    step 7/7 - loss: 0.0036 - 12s/step
    Eval begin...
    The loss value printed in the log is the current batch, and the metric is the average value of previous step.
    step 2/2 - loss: 7.2478e-04 - 4s/step
    Eval samples: 428


6. 模型预测
-----------

为了更好的观察预测结果，我们分别可视化验证集结果与标注点的对比，和在未标注的测试集的预测结果。
### 6.1 验证集结果可视化 红色的关键点为网络预测的结果，
绿色的关键点为标注的groundtrue。

.. code:: ipython3

    result = model.predict(val_dataset, batch_size=1)


.. parsed-literal::

    Predict begin...
    step 428/428 [==============================] - 29ms/step          
    Predict samples: 428


.. code:: ipython3

    def plot_sample(x, y, axis, gt=[]):
        img = x.reshape(96, 96)
        axis.imshow(img, cmap='gray')
        axis.scatter(y[0::2], y[1::2], marker='x', s=10, color='r')
        if gt!=[]:
            axis.scatter(gt[0::2], gt[1::2], marker='x', s=10, color='lime')
    
    
    fig = plt.figure(figsize=(10, 7))
    fig.subplots_adjust(
        left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
    
    for i in range(16):
        axis = fig.add_subplot(4, 4, i+1, xticks=[], yticks=[])
        idx = np.random.randint(val_dataset.__len__())
        img, gt_label = val_dataset[idx]
        gt_label = gt_label*96
        label_pred = result[0][idx].reshape(-1)
        label_pred = label_pred*96
        plot_sample(img[0], label_pred, axis, gt_label)
    plt.show()


.. parsed-literal::

    <ipython-input-19-0617ba14cacb>:5: DeprecationWarning: elementwise comparison failed; this will raise an error in the future.
      if gt!=[]:



.. image:: landmark_detection_files/landmark_detection_17_1.png


6.2 测试集结果可视化
~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    result = model.predict(test_dataset, batch_size=1)


.. parsed-literal::

    Predict begin...
    step 1783/1783 [==============================] - 28ms/step          
    Predict samples: 1783


.. code:: ipython3

    fig = plt.figure(figsize=(10, 7))
    fig.subplots_adjust(
        left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
    
    for i in range(16):
        axis = fig.add_subplot(4, 4, i+1, xticks=[], yticks=[])
        idx = np.random.randint(test_dataset.__len__())
        img, _ = test_dataset[idx]
        label_pred = result[0][idx].reshape(-1)
        label_pred = label_pred*96
        plot_sample(img[0], label_pred, axis)
    plt.show()



.. image:: landmark_detection_files/landmark_detection_20_0.png


