通过OCR实现验证码识别
=====================

| 作者: `GT_老张 <https://github.com/GT-ZhangAcer>`__
| 时间: 2020.11

| 本篇将介绍如何通过飞桨实现简单的CRNN+CTC自定义数据集OCR识别模型，数据集采用\ `CaptchaDataset <https://github.com/GT-ZhangAcer/CaptchaDataset>`__\ 中OCR部分的9453张图像，其中前8453张图像在本案例中作为训练集，后1000张则作为测试集。
| 在更复杂的场景中推荐使用\ `PaddleOCR <https://github.com/PaddlePaddle/PaddleOCR>`__\ 产出工业级模型，模型轻量且精度大幅提升。
| 同样也可以在\ `PaddleHub <https://www.paddlepaddle.org.cn/hubdetail?name=chinese_ocr_db_crnn_mobile&en_category=TextRecognition>`__\ 中快速使用PaddleOCR。

**数据展示**

.. image:: https://github.com/PaddlePaddle/FluidDoc/blob/develop/doc/paddle/tutorial/cv_case/image_ocr/OCR_files/OCR_01.png?raw=true

自定义数据集读取器
------------------

常见的开发任务中，我们并不一定会拿到标准的数据格式，好在我们可以通过自定义Reader的形式来随心所欲读取自己想要数据。

| 设计合理的Reader往往可以带来更好的性能，我们可以将读取标签文件列表、制作图像文件列表等必要操作在\ ``__init__``\ 特殊方法中实现。这样就可以在实例化\ ``Reader``\ 时装入内存，避免使用时频繁读取导致增加额外开销。同样我们可以在\ ``__getitem__``\ 特殊方法中实现如图像增强、归一化等个性操作，完成数据读取后即可释放该部分内存。
| 需要我们注意的是，如果不能保证自己数据十分纯净，可以通过\ ``try``\ 和\ ``expect``\ 来捕获异常并指出该数据的位置。当然也可以制定一个策略，使其在发生数据读取异常后依旧可以正常进行训练。

.. code:: ipython3

    import os
    
    import PIL.Image as Image
    import numpy as np
    from paddle.io import Dataset
    
    # 图片信息配置 - 通道数、高度、宽度
    IMAGE_SHAPE_C = 3
    IMAGE_SHAPE_H = 30
    IMAGE_SHAPE_W = 70
    # 数据集图片中标签长度最大值设置 - 因图片中均为4个字符，故该处填写为4即可
    LABEL_MAX_LEN = 4
    
    
    class Reader(Dataset):
        def __init__(self, data_path: str, is_val: bool = False):
            """
            数据读取Reader
            :param data_path: Dataset路径
            :param is_val: 是否为验证集
            """
            super().__init__()
            self.data_path = data_path
            # 读取Label字典
            with open(os.path.join(self.data_path, "label_dict.txt"), "r", encoding="utf-8") as f:
                self.info = eval(f.read())
            # 获取文件名列表
            self.img_paths = [img_name for img_name in self.info]
            # 将数据集后1000张图片设置为验证集，当is_val为真时img_path切换为后1000张
            self.img_paths = self.img_paths[-1000:] if is_val else self.img_paths[:-1000]
    
        def __getitem__(self, index):
            # 获取第index个文件的文件名以及其所在路径
            file_name = self.img_paths[index]
            file_path = os.path.join(self.data_path, file_name)
            # 捕获异常 - 在发生异常时终止训练
            try:
                # 使用Pillow来读取图像数据
                img = Image.open(file_path)
                # 转为Numpy的array格式并整体除以255进行归一化
                img = np.array(img, dtype="float32").reshape((IMAGE_SHAPE_C, IMAGE_SHAPE_H, IMAGE_SHAPE_W)) / 255
            except Exception as e:
                raise Exception(file_name + "\t文件打开失败，请检查路径是否准确以及图像文件完整性，报错信息如下:\n" + str(e))
            # 读取该图像文件对应的Label字符串，并进行处理
            label = self.info[file_name]
            label = list(label)
            # 将label转化为Numpy的array格式
            label = np.array(label, dtype="int32").reshape(LABEL_MAX_LEN)
    
            return img, label
    
        def __len__(self):
            # 返回每个Epoch中图片数量
            return len(self.img_paths)

模型配置
--------

定义模型结构以及模型输入
------------------------

模型方面使用的简单的CRNN-CTC结构，输入形为CHW的图像在经过CNN->Flatten->Linear->RNN->Linear后输出图像中每个位置所对应的字符概率。考虑到CTC解码器在面对图像中元素数量不一、相邻元素重复时会存在无法正确对齐等情况，故额外添加一个类别代表“分隔符”进行改善。

CTC相关论文：\ `Connectionist Temporal Classification: Labelling
Unsegmented Sequence Data with Recurrent
Neu <http://people.idsia.ch/~santiago/papers/icml2006.pdf>`__

.. image:: https://github.com/PaddlePaddle/FluidDoc/blob/develop/doc/paddle/tutorial/cv_case/image_ocr/OCR_files/OCR_02.png?raw=true

网络部分，因本篇采用数据集较为简单且图像尺寸较小并不适合较深层次网络。若在对尺寸较大的图像进行模型构建，可以考虑使用更深层次网络/注意力机制来完成。当然也可以通过目标检测形式先检出文本位置，然后进行OCR部分模型构建。

PaddleOCR 效果图如下：

.. image:: https://github.com/PaddlePaddle/FluidDoc/blob/develop/doc/paddle/tutorial/cv_case/image_ocr/OCR_files/OCR_03.png?raw=true


.. code:: ipython3

    import paddle
    
    # 分类数量设置 - 因数据集中共包含0~9共10种数字+分隔符，所以是11分类任务
    CLASSIFY_NUM = 11
    
    # 定义输入层，shape中第0维使用-1则可以在预测时自由调节batch size
    input_define = paddle.static.InputSpec(shape=[-1, IMAGE_SHAPE_C, IMAGE_SHAPE_H, IMAGE_SHAPE_W],
                                       dtype="float32",
                                       name="img")
    
    # 定义网络结构
    class Net(paddle.nn.Layer):
        def __init__(self, is_infer: bool = False):
            super().__init__()
            self.is_infer = is_infer
    
            # 定义一层3x3卷积+BatchNorm
            self.conv1 = paddle.nn.Conv2D(in_channels=IMAGE_SHAPE_C,
                                      out_channels=32,
                                      kernel_size=3)
            self.bn1 = paddle.nn.BatchNorm2D(32)
            # 定义一层步长为2的3x3卷积进行下采样+BatchNorm
            self.conv2 = paddle.nn.Conv2D(in_channels=32,
                                      out_channels=64,
                                      kernel_size=3,
                                      stride=2)
            self.bn2 = paddle.nn.BatchNorm2D(64)
            # 定义一层1x1卷积压缩通道数，输出通道数设置为比LABEL_MAX_LEN稍大的定值可获取更优效果，当然也可设置为LABEL_MAX_LEN
            self.conv3 = paddle.nn.Conv2D(in_channels=64,
                                      out_channels=LABEL_MAX_LEN + 4,
                                      kernel_size=1)
            # 定义全连接层，压缩并提取特征（可选）
            self.linear = paddle.nn.Linear(in_features=429,
                                       out_features=128)
            # 定义RNN层来更好提取序列特征，此处为双向LSTM输出为2 x hidden_size，可尝试换成GRU等RNN结构
            self.lstm = paddle.nn.LSTM(input_size=128,
                                   hidden_size=64,
                                   direction="bidirectional")
            # 定义输出层，输出大小为分类数
            self.linear2 = paddle.nn.Linear(in_features=64 * 2,
                                        out_features=CLASSIFY_NUM)
    
        def forward(self, ipt):
            # 卷积 + ReLU + BN
            x = self.conv1(ipt)
            x = paddle.nn.functional.relu(x)
            x = self.bn1(x)
            # 卷积 + ReLU + BN
            x = self.conv2(x)
            x = paddle.nn.functional.relu(x)
            x = self.bn2(x)
            # 卷积 + ReLU
            x = self.conv3(x)
            x = paddle.nn.functional.relu(x)
            # 将3维特征转换为2维特征 - 此处可以使用reshape代替
            x = paddle.tensor.flatten(x, 2)
            # 全连接 + ReLU
            x = self.linear(x)
            x = paddle.nn.functional.relu(x)
            # 双向LSTM - [0]代表取双向结果，[1][0]代表forward结果,[1][1]代表backward结果，详细说明可在官方文档中搜索'LSTM'
            x = self.lstm(x)[0]
            # 输出层 - Shape = (Batch Size, Max label len, Signal) 
            x = self.linear2(x)
    
            # 在计算损失时ctc-loss会自动进行softmax，所以在预测模式中需额外做softmax获取标签概率
            if self.is_infer:
                # 输出层 - Shape = (Batch Size, Max label len, Prob) 
                x = paddle.nn.functional.softmax(x)
            return x

训练准备
--------

定义label输入以及超参数
~~~~~~~~~~~~~~~~~~~~~~~

监督训练需要定义label，预测则不需要该步骤。

.. code:: ipython3

    # 数据集路径设置
    DATA_PATH = "./data/OCR_Dataset"
    # 训练轮数
    EPOCH = 10
    # 每批次数据大小
    BATCH_SIZE = 16
    
    label_define = paddle.static.InputSpec(shape=[-1, LABEL_MAX_LEN],
                                        dtype="int32",
                                        name="label")

定义CTC Loss
~~~~~~~~~~~~

了解CTC解码器效果后，我们需要在训练中让模型尽可能接近这种类型输出形式，那么我们需要定义一个CTC
Loss来计算模型损失。不必担心，在飞桨框架中内置了多种Loss，无需手动复现即可完成损失计算。

使用文档：\ `CTCLoss <https://www.paddlepaddle.org.cn/documentation/docs/zh/2.0-beta/api/paddle/nn/functional/loss/ctc_loss_cn.html#ctc-loss>`__

.. code:: ipython3

    class CTCLoss(paddle.nn.Layer):
        def __init__(self):
            """
            定义CTCLoss
            """
            super().__init__()
    
        def forward(self, ipt, label):
            input_lengths = paddle.full(shape=[BATCH_SIZE, 1],fill_value=LABEL_MAX_LEN + 4,dtype= "int64")
            label_lengths = paddle.full(shape=[BATCH_SIZE, 1],fill_value=LABEL_MAX_LEN,dtype= "int64")
            # 按文档要求进行转换dim顺序
            ipt = paddle.tensor.transpose(ipt, [1, 0, 2])
            # 计算loss
            loss = paddle.nn.functional.ctc_loss(ipt, label, input_lengths, label_lengths, blank=10)
            return loss

实例化模型并配置优化策略
~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    # 实例化模型
    model = paddle.Model(Net(), inputs=input_define, labels=label_define)

.. code:: ipython3

    # 定义优化器
    optimizer = paddle.optimizer.Adam(learning_rate=0.0001, parameters=model.parameters())
    
    # 为模型配置运行环境并设置该优化策略
    model.prepare(optimizer=optimizer,
                    loss=CTCLoss())

开始训练
--------

.. code:: ipython3

    # 执行训练
    model.fit(train_data=Reader(DATA_PATH),
                eval_data=Reader(DATA_PATH, is_val=True),
                batch_size=BATCH_SIZE,
                epochs=EPOCH,
                save_dir="output/",
                save_freq=1,
                log_freq=100)


.. parsed-literal::

    Epoch 1/10
    step 100/529 - loss: 0.0191 - 66ms/step
    Eval begin...
    step 63/63 - loss: 0.0081 - 27ms/step
    Eval samples: 1000
    Epoch 2/10
    step 100/529 - loss: 0.0084 - 64ms/step
    Eval begin...
    step 63/63 - loss: 0.0075 - 27ms/step
    Eval samples: 1000
    ...
    Epoch 9/10
    step 100/529 - loss: 0.0025 - 66ms/step
    Eval begin...
    step 63/63 - loss: 0.0082 - 26ms/step
    Eval samples: 1000
    Epoch 10/10
    step 100/529 - loss: 0.0022 - 63ms/step
    Eval begin...
    step 63/63 - loss: 0.0099 - 28ms/step
    Eval samples: 1000


预测前准备
----------

像定义训练Reader一样定义预测Reader
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    # 与训练近似，但不包含Label
    class InferReader(Dataset):
        def __init__(self, dir_path=None, img_path=None):
            """
            数据读取Reader(预测)
            :param dir_path: 预测对应文件夹（二选一）
            :param img_path: 预测单张图片（二选一）
            """
            super().__init__()
            if dir_path:
                # 获取文件夹中所有图片路径
                self.img_names = [i for i in os.listdir(dir_path) if os.path.splitext(i)[1] == ".jpg"]
                self.img_paths = [os.path.join(dir_path, i) for i in self.img_names]
            elif img_path:
                self.img_names = [os.path.split(img_path)[1]]
                self.img_paths = [img_path]
            else:
                raise Exception("请指定需要预测的文件夹或对应图片路径")
    
        def get_names(self):
            """
            获取预测文件名顺序 
            """
            return self.img_names
    
        def __getitem__(self, index):
            # 获取图像路径
            file_path = self.img_paths[index]
            # 使用Pillow来读取图像数据并转成Numpy格式
            img = Image.open(file_path)
            img = np.array(img, dtype="float32").reshape((IMAGE_SHAPE_C, IMAGE_SHAPE_H, IMAGE_SHAPE_W)) / 255
            return img
    
        def __len__(self):
            return len(self.img_paths)


参数设置
~~~~~~~~

.. code:: ipython3

    # 待预测目录
    INFER_DATA_PATH = "./sample_img"
    # 训练后存档点路径 - final 代表最终训练所得模型
    CHECKPOINT_PATH = "./output/final.pdparams"
    # 每批次处理数量
    BATCH_SIZE = 32

展示待预测数据
~~~~~~~~~~~~~~

.. code:: ipython3

    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 10))
    sample_idxs = np.random.choice(50000, size=25, replace=False)
    
    for img_id, img_name in enumerate(os.listdir(INFER_DATA_PATH)):
        plt.subplot(1, 3, img_id + 1)
        plt.xticks([])
        plt.yticks([])
        im = Image.open(os.path.join(INFER_DATA_PATH, img_name))
        plt.imshow(im, cmap=plt.cm.binary)
        plt.xlabel("Img name: " + img_name)
    plt.show()



.. image:: https://github.com/PaddlePaddle/FluidDoc/blob/develop/doc/paddle/tutorial/cv_case/image_ocr/OCR_files/OCR_04.png?raw=true


开始预测
--------

   飞桨2.0 CTC Decoder
   相关API正在迁移中，暂时使用\ `第三方解码器 <https://github.com/awni/speech/blob/072bcf9ff510d814fbfcaad43b2883ecf8f60806/speech/models/ctc_decoder.py>`__\ 进行解码。

.. code:: ipython3

    
    from ctc import decode
    
    # 实例化预测模型
    model = paddle.Model(Net(is_infer=True), inputs=input_define)
    # 加载训练好的参数模型
    model.load(CHECKPOINT_PATH)
    # 设置运行环境
    model.prepare()
    
    # 加载预测Reader
    infer_reader = InferReader(INFER_DATA_PATH)
    img_names = infer_reader.get_names()
    results = model.predict(infer_reader, batch_size=BATCH_SIZE)
    index = 0
    for result in results[0]:
        for prob in result:
            out, _ = decode(prob, blank=10)
            print(f"文件名：{img_names[index]}，预测结果为：{out}")
            index += 1


.. parsed-literal::

    Predict begin...
    step 1/1 [==============================] - 24ms/step
    Predict samples: 3
    文件名：9451.jpg，预测结果为：(3, 4, 6, 3)
    文件名：9450.jpg，预测结果为：(8, 2, 0, 5)
    文件名：9452.jpg，预测结果为：(0, 3, 0, 0)

