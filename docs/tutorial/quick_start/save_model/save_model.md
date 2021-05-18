# 模型保存及加载

**作者:** [PaddlePaddle](https://github.com/PaddlePaddle) <br>
**日期:** 2021.05 <br>
**摘要:** 本教程将基于Paddle高阶API对模型参数的保存和加载进行讲解。

## 一、简介
在日常训练模型过程中会遇到一些突发情况，导致训练过程主动或被动的中断，因此在模型没有完全训练好的情况下，需要高频的保存下模型参数，在发生意外时可以快速载入保存的参数继续训练；抑或是模型已经训练好了，需要使用训练好的参数进行预测或部署模型上线。面对上述情况，Paddle中提供了保存模型和提取模型的方法，支持从上一次保存状态开始训练，只要随时保存训练过程中的模型状态，就不用从初始状态重新训练。
下面将基于手写数字识别的模型讲解paddle如何保存及加载模型，并恢复训练，网络结构部分的讲解省略。

## 二、环境配置

本教程基于Paddle 2.1 编写，如果你的环境不是本版本，请先参考官网[安装](https://www.paddlepaddle.org.cn/install/quick) Paddle 2.1 。


```python
import paddle
import paddle.nn.functional as F
from paddle.nn import Layer
from paddle.vision.datasets import MNIST
from paddle.metric import Accuracy
from paddle.nn import Conv2D,MaxPool2D,Linear
from paddle.static import InputSpec
from paddle.vision.transforms import ToTensor

print(paddle.__version__)
```

    2.1.0


## 三、数据集
手写数字的MNIST数据集，包含60,000个用于训练的示例和10,000个用于测试的示例。这些数字已经过尺寸标准化并位于图像中心，图像是固定大小(28x28像素)，其值为0到1。该数据集的官方地址为：http://yann.lecun.com/exdb/mnist/
本例中使用飞桨自带的mnist数据集。使用from paddle.vision.datasets import MNIST 引入即可。


```python
train_dataset = MNIST(mode='train', transform=ToTensor())
test_dataset = MNIST(mode='test', transform=ToTensor())
```

    Cache file /home/aistudio/.cache/paddle/dataset/mnist/train-images-idx3-ubyte.gz not found, downloading https://dataset.bj.bcebos.com/mnist/train-images-idx3-ubyte.gz 
    Begin to download
    
    Download finished
    Cache file /home/aistudio/.cache/paddle/dataset/mnist/train-labels-idx1-ubyte.gz not found, downloading https://dataset.bj.bcebos.com/mnist/train-labels-idx1-ubyte.gz 
    Begin to download
    ........
    Download finished
    Cache file /home/aistudio/.cache/paddle/dataset/mnist/t10k-images-idx3-ubyte.gz not found, downloading https://dataset.bj.bcebos.com/mnist/t10k-images-idx3-ubyte.gz 
    Begin to download
    
    Download finished
    Cache file /home/aistudio/.cache/paddle/dataset/mnist/t10k-labels-idx1-ubyte.gz not found, downloading https://dataset.bj.bcebos.com/mnist/t10k-labels-idx1-ubyte.gz 
    Begin to download
    ..
    Download finished


## 四、模型组建


```python
class MyModel(Layer):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = paddle.nn.Conv2D(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2)
        self.max_pool1 = MaxPool2D(kernel_size=2, stride=2)
        self.conv2 = Conv2D(in_channels=6, out_channels=16, kernel_size=5, stride=1)
        self.max_pool2 = MaxPool2D(kernel_size=2, stride=2)
        self.linear1 = Linear(in_features=16*5*5, out_features=120)
        self.linear2 = Linear(in_features=120, out_features=84)
        self.linear3 = Linear(in_features=84, out_features=10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.max_pool1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.max_pool2(x)
        x = paddle.flatten(x, start_axis=1, stop_axis=-1)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear3(x)
        return x
```

## 五、模型训练
通过`Model` 构建实例，快速完成模型训练


```python
inputs = InputSpec([None, 784], 'float32', 'x')
labels = InputSpec([None, 10], 'float32', 'x')
model = paddle.Model(MyModel(), inputs, labels)

optim = paddle.optimizer.Adam(learning_rate=0.001, parameters=model.parameters())

model.prepare(
    optim,
    paddle.nn.CrossEntropyLoss(),
    Accuracy()
    )
model.fit(train_dataset,
        test_dataset,
        epochs=3,
        batch_size=64,
        save_dir='mnist_checkpoint',
        verbose=1
        )
```

    The loss value printed in the log is the current step, and the metric is the average value of previous steps.
    Epoch 1/3
    step 938/938 [==============================] - loss: 0.0245 - acc: 0.9412 - 32ms/step         
    save checkpoint at /home/aistudio/mnist_checkpoint/0
    Eval begin...
    step 157/157 [==============================] - loss: 0.0044 - acc: 0.9775 - 27ms/step        
    Eval samples: 10000
    Epoch 2/3
    step 938/938 [==============================] - loss: 0.0058 - acc: 0.9798 - 31ms/step         
    save checkpoint at /home/aistudio/mnist_checkpoint/1
    Eval begin...
    step 157/157 [==============================] - loss: 4.7741e-04 - acc: 0.9836 - 25ms/step    
    Eval samples: 10000
    Epoch 3/3
    step 938/938 [==============================] - loss: 0.0122 - acc: 0.9865 - 32ms/step         
    save checkpoint at /home/aistudio/mnist_checkpoint/2
    Eval begin...
    step 157/157 [==============================] - loss: 9.9224e-04 - acc: 0.9875 - 26ms/step      
    Eval samples: 10000
    save checkpoint at /home/aistudio/mnist_checkpoint/final


## 六、保存模型参数

目前Paddle框架有三种保存模型参数的体系，分别是：
#### paddle 高阶API-模型参数保存
    * paddle.Model.fit
    * paddle.Model.save
#### paddle 基础框架-动态图-模型参数保存 
    * paddle.save
#### paddle 基础框架-静态图-模型参数保存 
    * paddle.static.save
    * paddle.static.save_inference_model

下面将基于高阶API对模型保存与加载的方法进行讲解。


#### 方法一：
* paddle.Model.fit(train_data, epochs, batch_size, save_dir, log_freq) <br><br>
在使用model.fit函数进行网络循环训练时，在save_dir参数中指定保存模型的路径，save_freq指定写入频率，即可同时实现模型的训练和保存。mode.fit()只能保存模型参数，不能保存优化器参数，每个epoch结束只会生成一个.pdparams文件。可以边训练边保存，每次epoch结束会实时生成一个.pdparams文件。 

#### 方法二：
* paddle.Model.save(self, path, training=True) <br><br>
model.save(path)方法可以保存模型结构、网络参数和优化器参数，参数training=true的使用场景是在训练过程中，此时会保存网络参数和优化器参数。每个epoch生成两种文件 0.pdparams,0.pdopt，分别存储了模型参数和优化器参数，但是只会在整个模型训练完成后才会生成包含所有epoch参数的文件，path的格式为'dirname/file_prefix' 或 'file_prefix'，其中dirname指定路径名称，file_prefix 指定参数文件的名称。当training=false的时候，代表已经训练结束，此时存储的是预测模型结构和网络参数。


```python
# 方法一：训练过程中实时保存每个epoch的模型参数
model.fit(train_dataset,
        test_dataset,
        epochs=2,
        batch_size=64,
        save_dir='mnist_checkpoint',
        verbose=1
        )
```

    The loss value printed in the log is the current step, and the metric is the average value of previous steps.
    Epoch 1/2
    step 938/938 [==============================] - loss: 0.0024 - acc: 0.9886 - 32ms/step         
    save checkpoint at /home/aistudio/mnist_checkpoint/0
    Eval begin...
    step 157/157 [==============================] - loss: 2.9594e-04 - acc: 0.9876 - 27ms/step      
    Eval samples: 10000
    Epoch 2/2
    step 938/938 [==============================] - loss: 0.0013 - acc: 0.9915 - 31ms/step         
    save checkpoint at /home/aistudio/mnist_checkpoint/1
    Eval begin...
    step 157/157 [==============================] - loss: 5.1616e-05 - acc: 0.9891 - 26ms/step        
    Eval samples: 10000
    save checkpoint at /home/aistudio/mnist_checkpoint/final



```python
# 方法二：model.save()保存模型和优化器参数信息
model.save('mnist_checkpoint/test')
```

## 七、加载模型参数

当恢复训练状态时，需要加载模型数据，此时可以使用加载函数从存储模型状态和优化器状态的文件中载入模型参数和优化器参数，如果不需要恢复优化器，则不必使用优化器状态文件。
#### 高阶API-模型参数加载
    * paddle.Model.load
#### paddle 基础框架-动态图-模型参数加载
    * paddle.load
#### paddle 基础框架-静态图-模型参数加载
    * paddle.io.load 
    * paddle.io.load_inference_model

下面将对高阶API的模型参数加载方法进行讲解
* model.load(self, path, skip_mismatch=False, reset_optimizer=False)<br><br>
model.load能够同时加载模型和优化器参数。通过reset_optimizer参数来指定是否需要恢复优化器参数，若reset_optimizer参数为True，则重新初始化优化器参数，若reset_optimizer参数为False，则从路径中恢复优化器参数。


```python
# 高阶API加载模型
model.load('mnist_checkpoint/test')
```

## 八、恢复训练

理想的恢复训练是模型状态回到训练中断的时刻，恢复训练之后的梯度更新走向是和恢复训练前的梯度走向完全相同的。基于此，可以通过恢复训练后的损失变化，判断上述方法是否能准确的恢复训练。即从epoch 0结束时保存的模型参数和优化器状态恢复训练，校验其后训练的损失变化（epoch 1）是否和不中断时的训练完全一致。

说明：

恢复训练有如下两个要点：

* 保存模型时同时保存模型参数和优化器参数

* 恢复参数时同时恢复模型参数和优化器参数。


```python
import paddle
from paddle.vision.datasets import MNIST
from paddle.metric import Accuracy
from paddle.static import InputSpec

train_dataset = MNIST(mode='train', transform=ToTensor())
test_dataset = MNIST(mode='test', transform=ToTensor())

inputs = InputSpec([None, 784], 'float32', 'inputs')
labels = InputSpec([None, 10], 'float32', 'labels')
model = paddle.Model(MyModel(), inputs, labels)
optim = paddle.optimizer.Adam(learning_rate=0.001, parameters=model.parameters())
model.load("./mnist_checkpoint/final")
model.prepare( 
      optim,
      paddle.nn.loss.CrossEntropyLoss(),
      Accuracy()
      )
model.fit(train_data=train_dataset,
        eval_data=test_dataset,
        batch_size=64,
        epochs=2,
        verbose=1
        )
```

    The loss value printed in the log is the current step, and the metric is the average value of previous steps.
    Epoch 1/2
    step 360/938 [==========>...................] - loss: 0.0124 - acc: 0.9915 - ETA: 19s - 33ms/ste

## 九、总结


以上就是用Mnist手写数字识别的例子对保存模型、加载模型、恢复训练进行讲解，Paddle提供了很多保存和加载的API方法，您可以根据自己的需求进行选择。
