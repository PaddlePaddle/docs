# **申威下从源码编译**

## 环境准备

* **处理器：威鑫 3231(SW-WX3231)、威鑫 H8000(SW-WX H8000)**
* **操作系统：openEuler22.03、UOS V20、 Kylin V10**
* **Python 版本 3.9/3.11 (64 bit)**
* **pip 或 pip3 版本 9.0.1+ (64 bit)**

## 安装步骤

本文在申威处理器下安装 Paddle，接下来详细介绍各个步骤。

<a name="sw_source"></a>
### **源码编译**

1. 将 Paddle 的源代码克隆到当下目录下的 Paddle 文件夹中，并进入 Paddle 目录

    ```
    git clone https://github.com/PaddlePaddle/Paddle.git
    ```

    ```
    cd Paddle
    ```

2. 切换到`develop`分支下进行编译：

    ```
    git checkout develop
    ```

3. Paddle 依赖 cmake 进行编译构建，需要 cmake 版本>=3.15，检查操作系统源提供 cmake 的版本，使用源的方式直接安装 cmake, `apt install cmake`或`yum install cmake`, 检查 cmake 版本, `cmake --version`, 如果 cmake >= 3.15 则不需要额外的操作，否则请修改 Paddle 主目录的`CMakeLists.txt`, `cmake_minimum_required(VERSION 3.15)` 修改为 `cmake_minimum_required(VERSION 3.0)`.

4. 申威支持 openblas，使用 `yum` 安装 openblas 及其相关的依赖（如果安装失败，需要联系操作系统厂商解决安装问题）。
   安装 openblas，得到 openblas 库文件及头文件 cblas.h；
   ```
   yum install openblas-devel.sw_64
   ```

   编译时出现以下 log 信息，表明 openblas 库链接成功：
   ```
   -- Found OpenBLAS (include: /usr/include/openblas, library: /usr/lib/libopenblas.so)
   -- Found lapack in OpenBLAS (include: /usr/include)
   ```

5. 根据[requirements.txt](https://github.com/PaddlePaddle/Paddle/blob/develop/python/requirements.txt)安装 Python 依赖库，注意在申威系统中一般无法直接使用公网 pypi 仓库安装依赖包，建议从申威社区(https://developer.wxiat.cn/)获取 pypi 源安装，如果遇到部分依赖包无法安装的情况，请联系申威社区(https://developer.wxiat.cn/#/support/feedback)提供支持。此外也可以通过 pip 安装的时候加--no-deps 的方式来避免依赖包的安装，但该种方式可能导致包由于缺少依赖不可用。

6. 请创建并进入一个叫 build 的目录下：

    ```
    mkdir build && cd build
    ```

7. 链接过程中打开文件数较多，可能超过系统默认限制导致编译出错，设置进程允许打开的最大文件数：

    ```
    ulimit -n 81920
    ```

8. 执行 cmake：

    >具体编译选项含义请参见[编译选项表](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/install/Tables.html#Compile)

    For Python3.9:
    ```
    cmake .. -DPY_VERSION=3.9 -DPYTHON_EXECUTABLE=`which python3` -DWITH_MKL=OFF -DWITH_TESTING=OFF -DCMAKE_BUILD_TYPE=Release -DON_INFER=ON -DWITH_PYTHON=ON -DWITH_XBYAK=OFF -DWITH_SW=ON -DCMAKE_CXX_FLAGS="-Wno-error -w" -DWITH_RCCL=OFF
    ```

9. 编译。

    ```
    make -j$(nproc)
    ```

10. 编译成功后进入`Paddle/build/python/dist`目录下找到生成的`.whl`包。

11. 在当前机器或目标机器安装编译好的`.whl`包：

    ```
    python3 -m pip install -U（whl 包的名字）
    ```

恭喜，至此您已完成 PaddlePaddle 在 申威 环境下的编译安装。

## **验证安装**
安装完成后您可以使用 `python` 或 `python3` 进入 python 解释器，输入`import paddle` ，再输入
 `paddle.utils.run_check()`

如果出现`PaddlePaddle is installed successfully!`，说明您已成功安装。

使用 PaddlePaddle 构建和训练一个简单的多层感知机(MLP)来识别手写数字(MNIST 数据集)创建一个小型的神经网络模型，来验证 Paddle 是否已经正确安装。

```
vim mnist_mlp_example.py
```
```
import paddle
import paddle.nn.functional as F
from paddle.vision.transforms import ToTensor

# 检查 PaddlePaddle 是否安装成功
paddle.utils.run_check()

# 定义一个简单的多层感知机模型
class SimpleMLP(paddle.nn.Layer):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.fc1 = paddle.nn.Linear(784, 512)
        self.fc2 = paddle.nn.Linear(512, 10)

    def forward(self, inputs):
        x = paddle.flatten(inputs, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 准备数据集，指定本地路径
train_dataset = paddle.vision.datasets.MNIST(mode='train', transform=ToTensor())
test_dataset = paddle.vision.datasets.MNIST(mode='test', transform=ToTensor())

# 设置训练数据加载器
train_loader = paddle.io.DataLoader(train_dataset, batch_size=64, shuffle=True)

# 初始化模型和优化器
model = SimpleMLP()
optim = paddle.optimizer.Adam(parameters=model.parameters())

# 训练模型
model.train()
for epoch in range(1):
    for batch_id, data in enumerate(train_loader()):
        x_data, y_data = data
        logits = model(x_data)
        loss = F.cross_entropy(logits, y_data)
        if batch_id % 100 == 0:
            print(f"Epoch {epoch}, Batch {batch_id}, Loss {loss.numpy()}")
        loss.backward()
        optim.step()
        optim.clear_grad()

# 评估模型
model.eval()
accs = []
for batch_id, data in enumerate(paddle.io.DataLoader(test_dataset, batch_size=64)):
    x_data, y_data = data
    logits = model(x_data)
    acc = paddle.metric.accuracy(logits, y_data)
    accs.append(acc.numpy())
print(f"Test Accuracy: {sum(accs)/len(accs)}")
```
执行：
```
python3 mnist_mlp_example.py
```
正确输出应为：
```
Running verify PaddlePaddle program ...
I0202 12:38:40.496767 184765 pir_interpreter.cc:1541] New Executor is Running ...
I0202 12:38:40.500195 184765 pir_interpreter.cc:1564] pir interpreter is running by multi-thread mode ...
PaddlePaddle works well on 1 CPU.
PaddlePaddle is installed successfully! Let's start deep learning with PaddlePaddle now.
Cache file /root/.cache/paddle/dataset/mnist/train-images-idx3-ubyte.gz not found, downloading https://dataset.bj.bcebos.com/mnist/train-images-idx3-ubyte.gz
Begin to download
item 2421/2421 [============================>.] - ETA: 4.548712805504397e-05s - 2ms/item
Download finished
Cache file /root/.cache/paddle/dataset/mnist/train-labels-idx1-ubyte.gz not found, downloading https://dataset.bj.bcebos.com/mnist/train-labels-idx1-ubyte.gz
Begin to download
item 8/8 [============================>.] - ETA: 0.00018048437050310895s - 4ms/item
Download finished
Cache file /root/.cache/paddle/dataset/mnist/t10k-images-idx3-ubyte.gz not found, downloading https://dataset.bj.bcebos.com/mnist/t10k-images-idx3-ubyte.gz
Begin to download
item 403/403 [============================>.] - ETA: 0.0007260442008299027s - 1ms/item
Download finished
Cache file /root/.cache/paddle/dataset/mnist/t10k-labels-idx1-ubyte.gz not found, downloading https://dataset.bj.bcebos.com/mnist/t10k-labels-idx1-ubyte.gz
Begin to download
item 2/2 [===========================>..] - ETA: 4.3717678636312485e-05s - 401us/item
Download finished
Epoch 0, Batch 0, Loss 2.544161081314087
Epoch 0, Batch 100, Loss 0.22368277609348297
Epoch 0, Batch 200, Loss 0.2068479061126709
Epoch 0, Batch 300, Loss 0.18411417305469513
Epoch 0, Batch 400, Loss 0.25494638085365295
.....

```

## **如何卸载**
请使用以下命令卸载 PaddlePaddle：

```
python3 -m pip uninstall paddlepaddle
```
或
```
python3 -m pip uninstall paddlepaddle
```

## **备注**

如果您在使用过程中遇到编译失败等问题，请到 issue 或者(https://developer.wxiat.cn/#/support/feedback)中留言，我们会及时解决。
