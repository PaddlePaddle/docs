通过DCGAN实现人脸图像生成
=========================

| 作者:`ZMpursue <https://github.com/ZMpursue>`__
| 日期:2020.10.26

本教程将通过一个示例对DCGAN进行介绍。在向其展示许多真实人脸照片（数据集：\ `Celeb-A
Face <http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html>`__\ ）后，我们将训练一个生成对抗网络（GAN）来产生新人脸。本文将对该实现进行详尽的解释，并阐明此模型的工作方式和原因。并不需要过多专业知识，但是可能需要新手花一些时间来理解的模型训练的实际情况。为了节省时间，请尽量选择GPU进行训练。

1 简介
------

本项目基于paddlepaddle，结合生成对抗网络（DCGAN）,通过弱监督学习的方式，训练生成真实人脸照片

1.1 什么是GAN？
~~~~~~~~~~~~~~~

生成对抗网络（Generative Adversarial Network
[1]，简称GAN）是非监督式学习的一种方法，通过让两个神经网络相互博弈的方式进行学习。该方法最初由
lan·Goodfellow 等人于2014年提出，原论文见 `Generative Adversarial
Network <https://arxiv.org/abs/1406.2661>`__\ 。

生成对抗网络由一个生成网络与一个判别网络组成。生成网络从潜在空间（latent
space）中随机采样作为输入，其输出结果需要尽量模仿训练集中的真实样本。判别网络的输入为真实样本或生成网络的输出，其目的是将尽可能的分辨输入为真实样本或生成网络的输出。而生成网络则要尽可能地欺骗判别网络。两个网络相互对抗、不断调整参数。

让\ :math:`x`\ 是代表图像的数据。\ :math:`D(x)`\ 是判别器网络，输出的概率为\ :math:`x`\ 来自训练数据还是生成器。假设\ :math:`x`\ 为CHW格式，大小为3x64x64的图像数据，D为判别器网络，\ :math:`D(x)`\ 为\ :math:`𝑥`\ 来自训练数据还是生成器。当\ :math:`𝑥`\ 来自训练数据时\ :math:`𝐷(𝑥)`\ 尽量接近1，\ :math:`𝑥`\ 来自生成器时\ :math:`𝐷(𝑥)`\ 尽量接近0。
因此，\ :math:`𝐷(𝑥)`\ 也可以被认为是传统的二分类器。

对于生成器网络，
假设\ :math:`z`\ 为从标准正态分布采样的空间向量。\ :math:`G(z)`\ 表示生成器网络，该网络将矢量\ :math:`z`\ 映射到数据空间，\ :math:`G(z)`\ 表示生成器网络输出的图像。生成器的目标是拟合训练数据(:math:`p_{data}`)的分布，以便可以从该估计分布中生成假样本(:math:`p_g`)。

所以，\ :math:`D(G(z))`\ 是生成器\ :math:`G`\ 输出是真实的图像的概率。如Goodfellow的论文所述，\ :math:`D`\ 和\ :math:`G`\ 玩一个minmax游戏，其中\ :math:`D`\ 尝试最大化其正确分类真假的可能性\ :math:`logD(x)`\ ，以及\ :math:`G`\ 试图最小化以下可能性\ :math:`D`\ 会预测其输出是假的\ :math:`log(1-D(G(x)))`\ 。

GAN的损失函数可表示为：

   :math:`\underset{G}{\text{min}} \underset{D}{\text{max}}V(D,G) = \mathbb{E}_{x\sim p_{data}(x)}\big[logD(x)\big] + \mathbb{E}_{z\sim p_{z}(z)}\big[log(1-D(G(z)))\big]`

| 从理论上讲，此minmax游戏的解决方案是\ :math:`p_g = p_{data}`\ ，鉴别者会盲目猜测输入是真实的还是假的。但是，GAN的收敛理论仍在积极研究中，实际上GAN常常会遇到梯度消失/爆炸问题。
| 生成对抗网络常用于生成以假乱真的图片。此外，该方法还被用于生成视频、三维物体模型等。

1.2 什么是DCGAN？
~~~~~~~~~~~~~~~~~

DCGAN是深层卷积网络与GAN的结合，其基本原理与GAN相同，只是将生成网络和判别网络用两个卷积网络（CNN）替代。为了提高生成样本的质量和网络的收敛速度，论文中的
DCGAN 在网络结构上进行了一些改进：

-  取消 pooling 层：在网络中，所有的pooling层使用步幅卷积（strided
   convolutions）(判别器)和微步幅度卷积（fractional-strided
   convolutions）(生成器)进行替换。
-  加入batchnorm：在生成器和判别器中均加入batchnorm。
-  使用全卷积网络：去掉了FC层，以实现更深的网络结构。
-  激活函数：在生成器（G）中，最后一层使用Tanh函数，其余层采用ReLU函数 ;
   判别器（D）中都采用LeakyReLU。

2 环境设置及数据集
------------------

环境：paddlepaddle、scikit-image、numpy、matplotlib

| 在本教程中，我们将使用\ `Celeb-A
  Faces <http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html>`__\ 数据集，该数据集可以在链接的网站或\ `AI
  Studio <https://aistudio.baidu.com/aistudio/datasetdetail/39207>`__\ 中下载。数据集将下载为名为img_align_celeba.zip的文件。下载后，并将zip文件解压缩到该目录中。
| img_align_celeba目录结构应为：

::

   /path/to/project  
       -> img_align_celeba  
           -> 188242.jpg  
           -> 173822.jpg  
           -> 284702.jpg  
           -> 537394.jpg  
           ...

2.1 数据集预处理
~~~~~~~~~~~~~~~~

多线程处理，以裁切坐标(0,10)和(64,74)，将输入网络的图片裁切到64*64。

.. code:: ipython3

    from PIL import Image
    import os.path
    import os
    import threading
    from PIL import ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    
    '''多线程将图片缩放后再裁切到64*64分辨率'''
    #裁切图片宽度
    w = 64
    #裁切图片高度
    h = 64
    #裁切点横坐标(以图片左上角为原点)
    x = 0
    #裁切点纵坐标
    y = 20
    
    def cutArray(l, num):
      avg = len(l) / float(num)
      o = []
      last = 0.0
    
      while last < len(l):
        o.append(l[int(last):int(last + avg)])
        last += avg
    
      return o
      
    def convertjpg(jpgfile,outdir,width=w,height=h):
        img=Image.open(jpgfile)
        (l,h) = img.size
        rate = min(l,h) / width
        try:
            img = img.resize((int(l // rate),int(h // rate)),Image.BILINEAR)
            img = img.crop((x,y,width+x,height+y))
            img.save(os.path.join(outdir,os.path.basename(jpgfile)))
        except Exception as e:
            print(e)
    
    class thread(threading.Thread):
        def __init__(self, threadID, inpath, outpath, files):
            threading.Thread.__init__(self)
            self.threadID = threadID
            self.inpath = inpath
            self.outpath = outpath
            self.files = files
        def run(self):
            count = 0
            try:
                for file in self.files:
                    convertjpg(self.inpath + file,self.outpath)
                    count = count + 1
            except Exception as e:
                print(e)
            print('已处理图片数量：' +  str(count))
                
    if __name__ == '__main__':
        inpath = './work/img_align_celeba/'
        outpath = './work/imgs/'
        if not os.path.exists(outpath):
            os.mkdir(outpath)
        files =  os.listdir(inpath)
        files = cutArray(files,8)
        T1 = thread(1, inpath, outpath, files[0])
        T2 = thread(2, inpath, outpath, files[1])
        T3 = thread(3, inpath, outpath, files[2])
        T4 = thread(4, inpath, outpath, files[3])
        T5 = thread(5, inpath, outpath, files[4])
        T6 = thread(6, inpath, outpath, files[5])
        T7 = thread(7, inpath, outpath, files[6])
        T8 = thread(8, inpath, outpath, files[7])
        
        T1.start()
        T2.start()
        T3.start()
        T4.start()
        T5.start()
        T6.start()
        T7.start()
        T8.start()
        
        T1.join()
        T2.join()
        T3.join()
        T4.join()
        T5.join()
        T6.join()
        T7.join()
        T8.join()


3 模型组网
----------

3.1 定义数据预处理工具-Paddle.io.Dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

具体参考\ `Paddle.io.Dataset教程 <https://www.paddlepaddle.org.cn/documentation/docs/zh/2.0-rc/api/paddle/io/Dataset_cn.html#dataset>`__

.. code:: ipython3

    import os
    import cv2
    import numpy as np
    from skimage import io,color,transform
    import matplotlib.pyplot as plt
    import math
    import time
    import paddle
    from paddle.io import Dataset
    import six
    from PIL import Image as PilImage
    from paddle.static import InputSpec
    paddle.enable_static()
    img_dim = 64
    
    '''准备数据，定义Reader()'''
    PATH = 'work/imgs/'
    
    class DataGenerater(Dataset):
        """
        数据集定义
        """
        def __init__(self,path=PATH):
            """
            构造函数
            """
            super(DataGenerater, self).__init__()
            self.dir = path
            self.datalist = os.listdir(PATH)
            self.image_size = (img_dim,img_dim)
        
        # 每次迭代时返回数据和对应的标签
        def __getitem__(self, idx):
            return self._load_img(self.dir + self.datalist[idx])
    
        # 返回整个数据集的总数
        def __len__(self):
            return len(self.datalist)
        
        def _load_img(self, path):
            """
            统一的图像处理接口封装，用于规整图像大小和通道
            """
            try:
                img = io.imread(path)
                img = transform.resize(img,self.image_size)
                img = img.transpose()
                img = img.astype('float32')
            except Exception as e:
                    print(e)
            return img

3.2 测试Paddle.io.DataLoader并输出图片
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    
    train_dataset = DataGenerater()
    img = paddle.static.data(name='img', shape=[None,3,img_dim,img_dim], dtype='float32')
    train_loader = paddle.io.DataLoader(
        train_dataset, 
        places=paddle.CPUPlace(), 
        feed_list = [img],
        batch_size=128, 
        shuffle=True,
        num_workers=2,
        use_buffer_reader=True,
        use_shared_memory=False,
        drop_last=True,
        )
    
    for batch_id, data in enumerate(train_loader()):
        plt.figure(figsize=(15,15))
        try:
            for i in range(100):
                image = np.array(data[0]['img'][i])[0].transpose((2,1,0))
                plt.subplot(10, 10, i + 1)
                plt.imshow(image, vmin=-1, vmax=1)
                plt.axis('off')
                plt.xticks([])
                plt.yticks([])
                plt.subplots_adjust(wspace=0.1, hspace=0.1)
            plt.suptitle('\n Training Images',fontsize=30)
            plt.show()
            break
        except IOError:
            print(IOError)


3.3 权重初始化
~~~~~~~~~~~~~~

| 在 DCGAN
  论文中，作者指定所有模型权重应从均值为0、标准差为0.02的正态分布中随机初始化。
| 调用paddle.nn.initializer.Normal实现initialize设置

.. code:: ipython3

    conv_initializer=paddle.nn.initializer.Normal(mean=0.0, std=0.02)
    bn_initializer=paddle.nn.initializer.Normal(mean=1.0, std=0.02)

3.4 判别器
~~~~~~~~~~

如上文所述，生成器\ :math:`D`\ 是一个二进制分类网络，它以图像作为输入，输出图像是真实的（相对应\ :math:`G`\ 生成的假样本）的概率。输入\ :math:`Shape`\ 为[3,64,64]的RGB图像，通过一系列的\ :math:`Conv2d`\ ，\ :math:`BatchNorm2d`\ 和\ :math:`LeakyReLU`\ 层对其进行处理，然后通过全连接层输出的神经元个数为2，对应两个标签的预测概率。

-  将BatchNorm批归一化中momentum参数设置为0.5
-  将判别器(D)激活函数leaky_relu的alpha参数设置为0.2

..

   | 输入: 为大小64x64的RGB三通道图片
   | 输出: 经过一层全连接层最后为shape为[batch_size,2]的Tensor

.. code:: ipython3

    import paddle
    import paddle.nn as nn
    import paddle.nn.functional as F
    
    class Discriminator(paddle.nn.Layer):
        def __init__(self):
            super(Discriminator, self).__init__()
            self.conv_1 = nn.Conv2D(
                3,64,4,2,1,
                bias_attr=False,weight_attr=paddle.ParamAttr(name="d_conv_weight_1_",initializer=conv_initializer)
                )
            self.conv_2 = nn.Conv2D(
                64,128,4,2,1,
                bias_attr=False,weight_attr=paddle.ParamAttr(name="d_conv_weight_2_",initializer=conv_initializer)
                )
            self.bn_2 = nn.BatchNorm2D(
                128,
                weight_attr=paddle.ParamAttr(name="d_2_bn_weight_",initializer=bn_initializer),momentum=0.8
                )
            self.conv_3 = nn.Conv2D(
                128,256,4,2,1,
                bias_attr=False,weight_attr=paddle.ParamAttr(name="d_conv_weight_3_",initializer=conv_initializer)
                )
            self.bn_3 = nn.BatchNorm2D(
                256,
                weight_attr=paddle.ParamAttr(name="d_3_bn_weight_",initializer=bn_initializer),momentum=0.8
                )
            self.conv_4 = nn.Conv2D(
                256,512,4,2,1,
                bias_attr=False,weight_attr=paddle.ParamAttr(name="d_conv_weight_4_",initializer=conv_initializer)
                )
            self.bn_4 = nn.BatchNorm2D(
                512,
                weight_attr=paddle.ParamAttr(name="d_4_bn_weight_",initializer=bn_initializer),momentum=0.8
                )
            self.conv_5 = nn.Conv2D(
                512,1,4,1,0,
                bias_attr=False,weight_attr=paddle.ParamAttr(name="d_conv_weight_5_",initializer=conv_initializer)
                )
        
        def forward(self, x):
            x = self.conv_1(x)
            x = F.leaky_relu(x,negative_slope=0.2)
            x = self.conv_2(x)
            x = self.bn_2(x)
            x = F.leaky_relu(x,negative_slope=0.2)
            x = self.conv_3(x)
            x = self.bn_3(x)
            x = F.leaky_relu(x,negative_slope=0.2)
            x = self.conv_4(x)
            x = self.bn_4(x)
            x = F.leaky_relu(x,negative_slope=0.2)
            x = self.conv_5(x)
            x = F.sigmoid(x)
            return x

3.5 生成器
~~~~~~~~~~

生成器\ :math:`G`\ 旨在映射潜在空间矢量\ :math:`z`\ 到数据空间。由于我们的数据是图像，因此转换\ :math:`z`\ 到数据空间意味着最终创建具有与训练图像相同大小[3,64,64]的RGB图像。在网络设计中，这是通过一系列二维卷积转置层来完成的，每个层都与\ :math:`BatchNorm`\ 层和\ :math:`ReLu`\ 激活函数。生成器的输出通过\ :math:`tanh`\ 函数输出，以使其返回到输入数据范围[−1,1]。值得注意的是，在卷积转置层之后存在\ :math:`BatchNorm`\ 函数，因为这是DCGAN论文的关键改进。这些层有助于训练过程中的梯度更好地流动。

| **生成器网络结构**
| |models|

-  将\ :math:`BatchNorm`\ 批归一化中\ :math:`momentum`\ 参数设置为0.5

..

   | 输入:Tensor的Shape为[batch_size,100]其中每个数值大小为0~1之间的float32随机数
   | 输出:3x64x64RGB三通道图片

.. |models| image:: ./images/models.png

.. code:: ipython3

    
    class Generator(paddle.nn.Layer):
        def __init__(self):
            super(Generator, self).__init__()
            self.conv_1 = nn.Conv2DTranspose(
                100,512,4,1,0,
                bias_attr=False,weight_attr=paddle.ParamAttr(name="g_dconv_weight_1_",initializer=conv_initializer)
                )
            self.bn_1 = nn.BatchNorm2D(
                512,
                weight_attr=paddle.ParamAttr(name="g_1_bn_weight_",initializer=bn_initializer),momentum=0.8
                )
            self.conv_2 = nn.Conv2DTranspose(
                512,256,4,2,1,
                bias_attr=False,weight_attr=paddle.ParamAttr(name="g_dconv_weight_2_",initializer=conv_initializer)
                )
            self.bn_2 = nn.BatchNorm2D(
                256,
                weight_attr=paddle.ParamAttr(name="g_2_bn_weight_",initializer=bn_initializer),momentum=0.8
                )
            self.conv_3 = nn.Conv2DTranspose(
                256,128,4,2,1,
                bias_attr=False,weight_attr=paddle.ParamAttr(name="g_dconv_weight_3_",initializer=conv_initializer)
                )
            self.bn_3 = nn.BatchNorm2D(
                128,
                weight_attr=paddle.ParamAttr(name="g_3_bn_weight_",initializer=bn_initializer),momentum=0.8
                )
            self.conv_4 = nn.Conv2DTranspose(
                128,64,4,2,1,
                bias_attr=False,weight_attr=paddle.ParamAttr(name="g_dconv_weight_4_",initializer=conv_initializer)
                )
            self.bn_4 = nn.BatchNorm2D(
                64,
                weight_attr=paddle.ParamAttr(name="g_4_bn_weight_",initializer=bn_initializer),momentum=0.8
                )
            self.conv_5 = nn.Conv2DTranspose(
                64,3,4,2,1,
                bias_attr=False,weight_attr=paddle.ParamAttr(name="g_dconv_weight_5_",initializer=conv_initializer)
                )
            self.tanh = paddle.nn.Tanh()
        
        def forward(self, x):
            x = self.conv_1(x)
            x = self.bn_1(x)
            x = F.relu(x)
            x = self.conv_2(x)
            x = self.bn_2(x)
            x = F.relu(x)
            x = self.conv_3(x)
            x = self.bn_3(x)
            x = F.relu(x)
            x = self.conv_4(x)
            x = self.bn_4(x)
            x = F.relu(x)
            x = self.conv_5(x)
            x = self.tanh(x)
            return x


3.6 损失函数
~~~~~~~~~~~~

选用BCELoss,公式如下:

:math:`Out = -1 * (label * log(input) + (1 - label) * log(1 - input))`

.. code:: ipython3

    ###损失函数
    loss = paddle.nn.BCELoss()

4 模型训练
----------

设置的超参数为： \* 学习率：0.0002 \* 输入图片长和宽：64 \* Epoch: 8 \*
Mini-Batch：128 \* 输入Tensor长度：100 \* Adam：Beta1：0.5，Beta2：0.999

训练过程中的每一次迭代，生成器和判别器分别设置自己的迭代次数。为了避免判别器快速收敛到0，本教程默认每迭代一次，训练一次判别器，两次生成器。

.. code:: ipython3

    import IPython.display as display
    import warnings
    import paddle.optimizer as optim
    warnings.filterwarnings('ignore')
    
    img_dim = 64
    lr = 0.0002
    epoch = 5
    output = "work/Output/"
    batch_size = 128
    G_DIMENSION = 100
    beta1=0.5
    beta2=0.999
    output_path = 'work/Output'
    device = paddle.set_device('gpu')
    paddle.disable_static(device)
    
    real_label = 1.
    fake_label = 0.
    
    netD = Discriminator()
    netG = Generator()
    optimizerD = optim.Adam(parameters=netD.parameters(), learning_rate=lr, beta1=beta1, beta2=beta2)
    optimizerG = optim.Adam(parameters=netG.parameters(), learning_rate=lr, beta1=beta1, beta2=beta2)
    
    ###训练过程
    losses = [[], []]
    #plt.ion()
    now = 0
    for pass_id in range(epoch):
        
        # enumerate()函数将一个可遍历的数据对象组合成一个序列列表
        for batch_id, data in enumerate(train_loader()):
            #训练判别器 
            optimizerD.clear_grad()
            real_cpu = data[0]
            label = paddle.full((batch_size,1,1,1),real_label,dtype='float32')
            output = netD(real_cpu)
            errD_real = loss(output,label)
            errD_real.backward()
            optimizerD.step()
            optimizerD.clear_grad()
    
            noise = paddle.randn([batch_size,G_DIMENSION,1,1],'float32')
            fake = netG(noise)
            label = paddle.full((batch_size,1,1,1),fake_label,dtype='float32')
            output = netD(fake.detach())
            errD_fake = loss(output,label)
            errD_fake.backward()
            optimizerD.step()
            optimizerD.clear_grad()
    
            errD = errD_real + errD_fake
            
            losses[0].append(errD.numpy()[0])
            ###训练生成器
            optimizerG.clear_grad()
            noise = paddle.randn([batch_size,G_DIMENSION,1,1],'float32')
            fake = netG(noise)
            label = paddle.full((batch_size,1,1,1),real_label,dtype=np.float32,)
            output = netD(fake)
            errG = loss(output,label)
            errG.backward()
            optimizerG.step()
            optimizerG.clear_grad()
            
            losses[1].append(errG.numpy()[0])
            if batch_id % 100 == 0:
                if not os.path.exists(output_path):
                    os.makedirs(output_path)
                # 每轮的生成结果
                generated_image = netG(noise).numpy()
                imgs = []
                plt.figure(figsize=(15,15))
                try:
                    for i in range(100):
                        image = generated_image[i].transpose()
                        image = np.where(image > 0, image, 0)
                        plt.subplot(10, 10, i + 1)
                        plt.imshow(image, vmin=-1, vmax=1)
                        plt.axis('off')
                        plt.xticks([])
                        plt.yticks([])
                        plt.subplots_adjust(wspace=0.1, hspace=0.1)
                    msg = 'Epoch ID={0} Batch ID={1} \n\n D-Loss={2} G-Loss={3}'.format(pass_id, batch_id, errD.numpy()[0], errG.numpy()[0])
                    plt.suptitle(msg,fontsize=20)
                    plt.draw()
                    plt.savefig('{}/{:04d}_{:04d}.png'.format(output_path, pass_id, batch_id),bbox_inches='tight')
                    plt.pause(0.01)
                    display.clear_output(wait=True)
                except IOError:
                    print(IOError)
        paddle.save(netG.state_dict(), "work/generator.params")
    
    plt.close()


.. code:: ipython3

    plt.figure(figsize=(15, 6))
    x = np.arange(len(losses[0]))
    plt.title('Generator and Discriminator Loss During Training')
    plt.xlabel('Number of Batch')
    plt.plot(x,np.array(losses[0]),label='D Loss')
    plt.plot(x,np.array(losses[1]),label='G Loss')
    plt.legend()
    plt.savefig('work/Generator and Discriminator Loss During Training.png')
    plt.show()

5 模型评估
----------

生成器\ :math:`G`\ 和判别器\ :math:`D`\ 的损失与迭代变化图
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. figure:: ./images/loss.png
   :alt: loss

   loss

对比真实人脸图像（图一）和生成人脸图像（图二）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

图一
^^^^

|face_image1| ### 图二 |face_image2|

.. |face_image1| image:: ./images/face_image1.jpeg
.. |face_image2| image:: ./images/face_image2.jpeg

6 模型预测
----------

输入随机数让生成器\ :math:`G`\ 生成随机人脸
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

生成的RGB三通道64*64的图片路径位于“worl/Generate/”

.. code:: ipython3

    device = paddle.set_device('gpu')
    paddle.disable_static(device)
    try:
        generate = Generator()
        state_dict = paddle.load("work/generator.params")
        generate.set_state_dict(state_dict)
        noise = paddle.randn([100,100,1,1],'float32')
        generated_image = generate(noise).numpy()
        for j in range(100):
            image = generated_image[j].transpose()
            plt.figure(figsize=(4,4))
            plt.imshow(image)
            plt.axis('off')
            plt.xticks([])
            plt.yticks([])
            plt.subplots_adjust(wspace=0.1, hspace=0.1)
            plt.savefig('work/Generate/generated_' + str(j + 1), bbox_inches='tight')
            plt.close()
    except IOError:
        print(IOError)

7 项目总结
----------

简单介绍了一下DCGAN的原理，通过对原项目的改进和优化，一步一步依次对生成器和判别器以及训练过程进行介绍。
DCGAN采用一个随机噪声向量作为输入，输入通过与CNN类似但是相反的结构，将输入放大成二维数据。采用这种结构的生成模型和CNN结构的判别模型，DCGAN在图片生成上可以达到相当可观的效果。本案例中，我们利用DCGAN生成了人脸照片，您可以尝试更换数据集生成符合个人需求的图片，或尝试修改网络结构观察不一样的生成效果。

8 参考文献
----------

[1] Goodfellow, Ian J.; Pouget-Abadie, Jean; Mirza, Mehdi; Xu, Bing;
Warde-Farley, David; Ozair, Sherjil; Courville, Aaron; Bengio, Yoshua.
Generative Adversarial Networks. 2014. arXiv:1406.2661 [stat.ML].

[2] Andrej Karpathy, Pieter Abbeel, Greg Brockman, Peter Chen, Vicki
Cheung, Rocky Duan, Ian Goodfellow, Durk Kingma, Jonathan Ho, Rein
Houthooft, Tim Salimans, John Schulman, Ilya Sutskever, And Wojciech
Zaremba, Generative Models, OpenAI, [April 7, 2016]

[3] alimans, Tim; Goodfellow, Ian; Zaremba, Wojciech; Cheung, Vicki;
Radford, Alec; Chen, Xi. Improved Techniques for Training GANs. 2016.
arXiv:1606.03498 [cs.LG].

[4] Radford A, Metz L, Chintala S. Unsupervised Representation Learning
with Deep Convolutional Generative Adversarial Networks[J]. Computer
Science, 2015.

[5]Kingma D , Ba J . Adam: A Method for Stochastic Optimization[J].
Computer ence, 2014.
