
# 1、任务描述：

    飞桨框架于 2.0 正式版全面支持了动态图训练，并在 2.1、2.2 两个大版本中不断完善分布式能力，同时大幅增强了训练功能。在本任务中，我们希望能收到你对于飞桨动态图分布式训练功能的使用感受，可以与其他深度学习框架做功能对比，并产出一份对应的评估报告。

# 2、环境配置：

    因为需要将飞桨 PaddlePaddle 框架的分布式与其他深度学习框架做功能对比，这里其他深度学习框架我选择了 PyTorch 框架，所以首先需要安装飞桨 PaddlePaddle 框架与 PyTorch 框架，开发平台选择了曙光平台昆山超算。

## 2.1、PyTorch 环境配置：

- 1、首先安装 anaconda
```bash
bash Anaconda3-2020.07-Linux-x86_64.sh –u
```
- 2、创建一个属于自己的环境并激活
```bash
conda create --name pytorch_1.9 python=3.7
conda activate pytorch-1.9
```
- 3、安装 pytorch-1.9（适配 rocm-4.0.1 及以上）PyTorch1.8 和 PyTorch1.9 安装 wheel 包在公共目录：
```bash
/public/software/apps/DeepLearning/whl/rocm-4.0.1/
```
- 安装指令如下：
```bash
module rm compiler/rocm/2.9
module load compiler/rocm/4.0.1
pip install /public/software/apps/DeepLearning/whl/rocm-4.0.1/torch-1.9.0+rocm4.0.1-cp36-cp36m-linux_x86_64.whl -i https://pypi.tuna.tsinghua.edu.cn/simple/
```
- 对于 torchverion 的安装不能按照曙光官方帮助文档给定的方法来，否则 torchvision 在运行自定义算子时会出现错误，所以需要使用源码安装的方式，安装方法如下:
```text
1、本地下载对应的 torchvision 分支源码包：https://github.com/pytorch/vision 上传集群，
2、进入对应的 conda 环境，加载对应的 rocm（这里 rocm4.0.1）版本；
3、conda install libpng -y
4、conda install jpeg -y
5、pip3 install numpy pillow matplotlib ninja -i https://pypi.tuna.tsinghua.edu.cn/simple/
6、使用 salloc 申请计算结点，使用 ssh 登录至计算节点，并进入对应的 conda 环境加载 rocm（这里 rocm4.0.1），执行编译：CC=clang CXX=clang++ python setup.py install
```

## 2.2、PaddlePaddle 环境配置：

- PaddlePaddle 的环境在曙光超算上需要使用镜像的方式进行安装，镜像添加，源镜像名称填：paddlepaddle/paddle，源镜像标签填：latest-dev-rocm4.0-miopen2.11。然后创建实例打开容器。因为 Docker 容器中不能连接网络，使用 paddle 官网给出的安装方式会出现网络连接的错误。
```bash
python -m pip install paddlepaddle-rocm==2.2.2.rocm401.miopen211 -f https://www.paddlepaddle.org.cn/whl/rocm/stable.whl
```
- 故需要提前下载 whl 文件，下载链接如下，下载的版本为 paddlepaddle_rocm-2.1.1.rocm401.miopen211-cp37-cp37m-linux_x86_64.whl。下载链接：

https://www.paddlepaddle.org.cn/whl/rocm/stable.whl

- 安装指令为
```bash
pip install paddlepaddle_rocm-2.1.1.rocm401.miopen211-cp37-cp37m-linux_x86_64.whl
```
- 期间所需要的其他库都需要通过在曙光超算上通过 EShell 进行安装，需要设定清华镜像源，例如
```bash
pip install six -i https://pypi.tuna.tsinghua.edu.cn/simple
```
- 经过测试，测试指令为
```bash
python -c "import paddle; paddle.utils.run_check()"
```
环境有效。

## 2.3、安装对比分析

PyTorch 的分布式环境在曙光平台安装时需要手动源码编译 torchversion，这个过程比较慢，这一点上 PyTorch 相对比较繁琐；但是 PyTorch 的环境在曙光平台比较稳定，而 PaddlePaddle 环境在曙光平台不太稳定。

# 3、Paddle 单机与分布式：

## 3.1、Paddle 单机

在图像处理中，关键点本质上是一种特征。它是对一个固定区域或者空间物理关系的抽象描述，描述的是一定邻域范围内的组合或上下文关系。它不仅仅是一个点信息，或代表一个位置，更代表着上下文与周围邻域的组合关系。关键点检测的目标就是通过计算机从图像中找出这些点的坐标，作为计算机视觉领域的一个基础任务，关键点的检测对于高级别任务，例如识别和分类具有至关重要的意义。任务选用的是人脸关键点检测，采用的方法是坐标点回归的方式进行，参考 PaddlePaddle 网址如下：
https://www.paddlepaddle.org.cn/documentation/docs/zh/practices/cv/landmark_detection.html

其思路如下：
- 1、导入相关库
- 2、构建数据集
- 3、定义模型
- 4、构建损失函数与优化器
- 5、训练模型
- 6、预测

具体分析如下：
- 第一步：导入相关库，即导入与本次任务相关的库，例如 import paddle
- 第二步：构建数据集，首先自定义一个处理人脸的 datasets，其继承于 Dataset 类，然后实现__init__,__getitem__与__len__函数，__init__函数实现一些数据集的初始化，即完成了对 csv 文件的读取与数据的清洗以及训练集、验证集、测试集的划分，__getitem__函数实现了通过 index 参数获取对应的 image 与 label，同时对图像进行预处理操作，即 transform 操作，__len__函数即返回数据集的数量（长度）。
- 第三步：定义模型，构建一个模型的类，继承于 paddle.nn.Layer，然后实现__init__与 forward 函数，__init__函数定义网络模型的一些层结构，forward 函数实现网络模型的前向传播。同时，示例中给出的代码中网络模型经过了 paddle.Model 类的封装，paddle.Model 类相对来说会更方便一些，其内部定义了一些 fit 等函数可以直接使用。
- 第四步：定义损失函数与优化器，优化器采用了 Adam 算法，同时初始学习率指定为 0.001，损失函数采用均方误差函数，因为网络模型经过了 paddle.Model 类的封装，所以此处对于损失函数与优化器需要使用 prepare 函数进行指定。
- 第五步：构建训练代码，网络模型经过了 paddle.Model 类的封装，训练只需要调用其内部的 fit 函数，对于 fit 函数，其内部会把传入的数据集进行 DataLoader 封装，同时通过 tain_batch 函数进行其训练过程。
- 第六步：测试，通过调用 paddle.Model 类的 predict 函数可以直接得到其预测的结果。
- 总结：上述单机的过程中，其采用了 paddle.Model 类的封装，会在使用方面方便很多，并且 paddle.Model 类的内部会在模型训练的过程中加入一些回调函数 callback 进行使用，相对来说比较好用一些。

## 3.2、paddle 分布式的实现

### 方式一：
首先对于上述代码进行分析，在 paddle.Model 的底层可以发现其在对模型进行初始化的时候会判断是动态图的模式还是静态图的模式，即
```python
if fluid.in_dygraph_mode():
    self._adapter = DynamicGraphAdapter(self)
else:
    self._adapter = StaticGraphAdapter(self)
```
因为我们采用的是动态图，所以其会调用第一个函数，即初始化 DynamicGraphAdapter(self)，观察其底层内部我们可以发现其会进行单机还是分布式的判断，如果是分布式，即 tasks>1，会初始化分布式的环境同时构建分布式的模型，代码分析如下：

```python
if self._nranks > 1:
    dist.init_parallel_env()
    stradegy = fluid.dygraph.parallel.ParallelStrategy()
    stradegy.nranks = ParallelEnv().nranks
    stradegy.local_rank = ParallelEnv().local_rank
    stradegy.trainer_endpoints = ParallelEnv().trainer_endpoints
    stradegy.current_endpoint = ParallelEnv().current_endpoint
    self.ddp_model = fluid.dygraph.parallel.DataParallel(
        self.model.network, stradegy)

```
这样的话，分布式的环境与分布式的模型就已经在 paddle.Model 内的内部构建完成了。
分析 model.prepare 即设置训练过程中的优化器与损失函数中我们发现其内部并没有设置分布式优化器，故这一部分需要自己在代码中添加，即
```python
optim = fleet.distributed_optimizer(optim)
model.prepare(optim, paddle.nn.MSELoss())
```
分析训练的过程，即 model.fit 的底层代码，我们可以发现其底层构建数据集是就采用了分布式的数据集划分，即其 sampler 采用的是 DistributedBatchSampler，所以其训练时也就采用的是分布式的训练过程，即把数据划分到各个设备上进行训练。

总结：相对来说，示例中的代码使用了 paddle.Model 的封装，而其 paddle.Model 底层的一些函数中也添加了对分布式的处理，所以从示例代码中改成分布式的代码只需要进行分布式优化器的构建，即将
```python
model.prepare(optim, paddle.nn.MSELoss())
```
改成
```python
optim = fleet.distributed_optimizer(optim)
model.prepare(optim, paddle.nn.MSELoss())
```
### 方式二：
paddle.Model 类内部封装的东西使用起来比较方便，但是不利于用户了解单机转成分布式的具体流程，所以我没有使用 paddle.Model 类，而是重新编写了分布式的代码。其流程如下：

- 1、导入分布式所需要的依赖包
- 2、初始化分布式环境
- 3、构建模型
- 4、设置分布式所需要的优化器
- 5、数据集的拆分
- 6、构建训练代码
- 7、启动分布式任务

下面具体流程如下：
- 第一步：导入分布式所需要的依赖包，即导入任务相关的 API 函数，例如
```python
from paddle.distributed import fleet
```
- 第二步：初始化分布式环境，采用了 collective 通信，代码如下:
```python
fleet.init(is_collective=True)
```
- 第三步：构建网络模型，这个网络模型采用了和示例单机中相同的网络模型。代码如下：
```python
model = FaceNet(num_keypoints=15)
model = fleet.distributed_model(model)
```
- 第四步：设置分布式所需要的优化器，优化器采用了 Adam 优化器，初始学习率为 0.001，代码如下：
```python
optim = paddle.optimizer.Adam(learning_rate=1e-3, parameters=model.parameters())
    optim = fleet.distributed_optimizer(optim)
```
- 第五步：数据集的拆分
对于分布式的数据拆分，需要先构建其数据集的采样器，这里需要使用 DistributedBatchSampler，其中参数为数据集 dataset、batch_size、num_replicas、rank、shuffle、drop_last，这里我指定了 dataset、batch_size、shuffle，设置了 shuffle 为 True，即对数据进行打乱，其中的参数 num_replicas 如果不指定，其默认会获取当前环境中的 ntasks，然后按照 ntasks 分配数据集。drop_last 参数如果不指定会默认为 False，也就是不会丢失最后一个 batch 的数据。构建完分布式采样器之后，使用 DataLoader 进行封装一下，这里指定一下 batch_sampler 为刚才构建的采样器，注意指定 batch_sampler 参数之后不需要再指定 batch_size、shuffle 以及 drop_last 参数。
- 第六步：构建训练代码
这里不采用 paddle.Model 进行封装，所以需要自己编写 for 循环获取数据进行前向传播以及反向传播的过程，代码如下：
```python
for eop in range(epoch):
    # train_sampler.set_epoch(eop)
    # 设置为训练模式
    model.train()
    for batch_id, data in enumerate(train_loader()):
        img, label = data
        label.stop_gradient = True
        # 前向传播
        out = model(img)
        # 均方损失函数
        loss = paddle.nn.functional.mse_loss(input=out, label=label)
        loss_data = loss.numpy()
        # 反向传播
        loss.backward()
        optim.step()
        model.clear_gradients()
        if batch_id % 10 == 0:
            print("[Epoch %d, batch %d] loss: %.5f" % (eop, batch_id, loss_data))
```
* 第七步：
飞桨通过 paddle.distributed.launch 组件启动分布式任务。该组件可用于启动单机多卡分布式任务，也可以用于启动多机多卡分布式任务。该组件为每张参与分布式任务的训练卡启动一个训练进程。默认情形下，该组件将在每个节点上启动 N 个进程，这里 N 等于训练节点的卡数，即使用所有的训练卡。用户也可以通过 gpus 参数指定训练节点上使用的训练卡列表，该列表以逗号分隔。需要注意的是，所有节点需要使用相同数量的训练卡数。为了启动多机分布式任务，需要通过 ips 参数指定所有节点的 IP 地址列表，该列表以逗号分隔。需要注意的是，该列表在所有节点上需要保持一致，即各节点 IP 地址出现的顺序需要保持一致。这里我进行了单机多卡与多机多卡的实验，实验启动方式如下：

单机多卡分布式任务：这里我采用的是四个卡，启动方式如下：
```python
python -m paddle.distributed.launch --gpus 0,1,2,3 train_multi_gpu.py
```
运行结果如下：
```python
-----------  Configuration Arguments -----------
gpus: 0,1,2,3
heter_worker_num: None
heter_workers:
http_port: None
ips: 127.0.0.1
log_dir: log
nproc_per_node: None
run_mode: None
server_num: None
servers:
training_script: train_multi_4.py
training_script_args: []
worker_num: None
workers:
------------------------------------------------
WARNING 2022-04-02 23:29:34,154 launch.py:359] Not found distinct arguments and compiled with cuda or xpu. Default use collective mode
launch train in GPU mode!
INFO 2022-04-02 23:29:34,156 launch_utils.py:510] Local start 4 processes. First process distributed environment info (Only For Debug):
    +=======================================================================================+
    |                        Distributed Envs                      Value                    |
    +---------------------------------------------------------------------------------------+
    |                       PADDLE_TRAINER_ID                        0                      |
    |                 PADDLE_CURRENT_ENDPOINT                 127.0.0.1:40344               |
    |                     PADDLE_TRAINERS_NUM                        4                      |
    |                PADDLE_TRAINER_ENDPOINTS  ... 0.1:41302,127.0.0.1:54290,127.0.0.1:60414|
    |                     PADDLE_RANK_IN_NODE                        0                      |
    |                 PADDLE_LOCAL_DEVICE_IDS                        0                      |
    |                 PADDLE_WORLD_DEVICE_IDS                     0,1,2,3                   |
    |                     FLAGS_selected_gpus                        0                      |
    |             FLAGS_selected_accelerators                        0                      |
    +=======================================================================================+

INFO 2022-04-02 23:29:34,156 launch_utils.py:514] details abouts PADDLE_TRAINER_ENDPOINTS can be found in log/endpoints.log, and detail running logs maybe found in log/workerlog.0
launch proc_id:22388 idx:0
launch proc_id:22391 idx:1
launch proc_id:22394 idx:2
launch proc_id:22397 idx:3
W0402 23:29:36.404554 22388 gen_comm_id_helper.cc:120] connect addr=127.0.0.1:41302 failed 1 times with reason: Connection refused retry after 0.5 seconds
I0402 23:29:36.905133 22388 nccl_context.cc:74] init nccl context nranks: 4 local rank: 0 gpu id: 0 ring id: 0
W0402 23:30:20.059859 22388 device_context.cc:404] Please NOTE: device: 0, GPU Compute Capability: 90.0, Driver API Version: 321.0, Runtime API Version: 3.1
W0402 23:30:20.067070 22388 device_context.cc:417] device: 0, MIOpen Version: 2.11.0
I0402 23:30:20.082139 22388 nccl_context.cc:107] init nccl context nranks: 4 local rank: 0 gpu id: 0 ring id: 10

/public/home/ac48p2il5w/anaconda3/envs/paddle_wjc_task/lib/python3.7/site-packages/paddle/tensor/creation.py:125: DeprecationWarning: `np.object` is a deprecated alias for the builtin `object`. To silence this warning, use `object` by itself. Doing this will not modify any behavior and is safe.
Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
  if data.dtype == np.object:
2022-04-02 23:32:10,565-INFO: [topology.py:152:__init__] HybridParallelInfo: rank_id: 0, dp_degree: 4, mp_degree: 1, pp_degree: 1, dp_group: [0, 1, 2, 3], mp_group: [0], pp_group: [0], check/clip group: [0]
Epoch  1 / 30
/public/home/ac48p2il5w/anaconda3/envs/paddle_wjc_task/lib/python3.7/site-packages/paddle/nn/layer/norm.py:641: UserWarning: When training, we now always track global mean and variance.
  "When training, we now always track global mean and variance.")
step 0 / 14 - loss: 0.91819
step 10 / 14 - loss: 0.06710
Eval begin...
step 0 / 4 - loss: 0.02265
Eval samples:  428
Epoch  2 / 30
step 0 / 14 - loss: 0.04950
step 10 / 14 - loss: 0.02599
Eval begin...
step 0 / 4 - loss: 0.00559
Eval samples:  428
Epoch  3 / 30
step 0 / 14 - loss: 0.02373
step 10 / 14 - loss: 0.01906
Eval begin...
step 0 / 4 - loss: 0.00270
Eval samples:  428
Epoch  4 / 30
step 0 / 14 - loss: 0.01631
step 10 / 14 - loss: 0.01318
Eval begin...
step 0 / 4 - loss: 0.00227
Eval samples:  428
Epoch  5 / 30
step 0 / 14 - loss: 0.01354
step 10 / 14 - loss: 0.01125
Eval begin...
step 0 / 4 - loss: 0.00136
Eval samples:  428
Epoch  6 / 30
step 0 / 14 - loss: 0.01162
step 10 / 14 - loss: 0.00890
Eval begin...
step 0 / 4 - loss: 0.00145
Eval samples:  428
Epoch  7 / 30
step 0 / 14 - loss: 0.01274
step 10 / 14 - loss: 0.01234
Eval begin...
step 0 / 4 - loss: 0.00259
Eval samples:  428
Epoch  8 / 30
step 0 / 14 - loss: 0.00965
step 10 / 14 - loss: 0.00664
Eval begin...
step 0 / 4 - loss: 0.00200
Eval samples:  428
Epoch  9 / 30
step 0 / 14 - loss: 0.00763
step 10 / 14 - loss: 0.00707
Eval begin...
step 0 / 4 - loss: 0.00129
Eval samples:  428
Epoch  10 / 30
step 0 / 14 - loss: 0.00697
step 10 / 14 - loss: 0.00591
Eval begin...
step 0 / 4 - loss: 0.00150
Eval samples:  428
Epoch  11 / 30
step 0 / 14 - loss: 0.00708
step 10 / 14 - loss: 0.00488
Eval begin...
step 0 / 4 - loss: 0.00376
Eval samples:  428
Epoch  12 / 30
step 0 / 14 - loss: 0.00859
step 10 / 14 - loss: 0.00602
Eval begin...
step 0 / 4 - loss: 0.00262
Eval samples:  428
Epoch  13 / 30
step 0 / 14 - loss: 0.00815
step 10 / 14 - loss: 0.00786
Eval begin...
step 0 / 4 - loss: 0.00187
Eval samples:  428
Epoch  14 / 30
step 0 / 14 - loss: 0.00540
step 10 / 14 - loss: 0.00839
Eval begin...
step 0 / 4 - loss: 0.00235
Eval samples:  428
Epoch  15 / 30
step 0 / 14 - loss: 0.00568
step 10 / 14 - loss: 0.00725
Eval begin...
step 0 / 4 - loss: 0.00189
Eval samples:  428
Epoch  16 / 30
step 0 / 14 - loss: 0.00657
step 10 / 14 - loss: 0.00842
Eval begin...
step 0 / 4 - loss: 0.00112
Eval samples:  428
Epoch  17 / 30
step 0 / 14 - loss: 0.00573
step 10 / 14 - loss: 0.00642
Eval begin...
step 0 / 4 - loss: 0.00297
Eval samples:  428
Epoch  18 / 30
step 0 / 14 - loss: 0.00617
step 10 / 14 - loss: 0.00632
Eval begin...
step 0 / 4 - loss: 0.00399
Eval samples:  428
Epoch  19 / 30
step 0 / 14 - loss: 0.00850
step 10 / 14 - loss: 0.00664
Eval begin...
step 0 / 4 - loss: 0.00083
Eval samples:  428
Epoch  20 / 30
step 0 / 14 - loss: 0.00555
step 10 / 14 - loss: 0.01062
Eval begin...
step 0 / 4 - loss: 0.00103
Eval samples:  428
Epoch  21 / 30
step 0 / 14 - loss: 0.00480
step 10 / 14 - loss: 0.00547
Eval begin...
step 0 / 4 - loss: 0.00085
Eval samples:  428
Epoch  22 / 30
step 0 / 14 - loss: 0.00463
step 10 / 14 - loss: 0.00429
Eval begin...
step 0 / 4 - loss: 0.00088
Eval samples:  428
Epoch  23 / 30
step 0 / 14 - loss: 0.00439
step 10 / 14 - loss: 0.00575
Eval begin...
step 0 / 4 - loss: 0.00302
Eval samples:  428
Epoch  24 / 30
step 0 / 14 - loss: 0.00447
step 10 / 14 - loss: 0.00492
Eval begin...
step 0 / 4 - loss: 0.00134
Eval samples:  428
Epoch  25 / 30
step 0 / 14 - loss: 0.00526
step 10 / 14 - loss: 0.00549
Eval begin...
step 0 / 4 - loss: 0.00141
Eval samples:  428
Epoch  26 / 30
step 0 / 14 - loss: 0.00387
step 10 / 14 - loss: 0.00477
Eval begin...
step 0 / 4 - loss: 0.00243
Eval samples:  428
Epoch  27 / 30
step 0 / 14 - loss: 0.00668
step 10 / 14 - loss: 0.00480
Eval begin...
step 0 / 4 - loss: 0.00094
Eval samples:  428
Epoch  28 / 30
step 0 / 14 - loss: 0.00388
step 10 / 14 - loss: 0.00392
Eval begin...
step 0 / 4 - loss: 0.00090
Eval samples:  428
Epoch  29 / 30
step 0 / 14 - loss: 0.00384
step 10 / 14 - loss: 0.00503
Eval begin...
step 0 / 4 - loss: 0.00099
Eval samples:  428
Epoch  30 / 30
step 0 / 14 - loss: 0.00454
step 10 / 14 - loss: 0.00398
Eval begin...
step 0 / 4 - loss: 0.00083
Eval samples:  428
INFO 2022-04-02 23:34:34,617 launch.py:268] Local processes completed.
```
多机多卡分布式任务：相对于单机多卡的分布式任务，多机多卡下不需要对代码有任何的更改，故不需要更改程序，只需要改变一下启动方式。这里对于多机的情况，我在曙光超算上开启了两个镜像，每个镜像申请了 2 个加速卡，开启之后首先使用 ifconfig 查看两个镜像下的 ip 地址，然后使用 ping 指令查看一下两个镜像能否相互 ping 通，然后分别在两个镜像下使用下面指令运行：（其中 ips 中的两个 ip 地址换成镜像中的 ip 地址）
```python
python -m paddle.distributed.launch --ips="192.168.0.1,192.168.0.2" --gpus 0,1 train_fleet_dygraph.py
```
运行结果如下：
```python
-----------  Configuration Arguments -----------
gpus: 0,1
heter_worker_num: None
heter_workers:
http_port: None
ips: 173.8.206.5,173.15.88.4
log_dir: log
nproc_per_node: None
run_mode: None
server_num: None
servers:
training_script: train_multi_4.py
training_script_args: []
worker_num: None
workers:
------------------------------------------------
INFO 2022-04-03 16:58:50,318 launch.py:348] Run collective mode. gpu arguments:['--ips'], cuda count:2
launch train in GPU mode!
INFO 2022-04-03 16:58:50,320 launch_utils.py:510] Local start 2 processes. First process distributed environment info (Only For Debug):
    +=======================================================================================+
    |                        Distributed Envs                      Value                    |
    +---------------------------------------------------------------------------------------+
    |                       PADDLE_TRAINER_ID                        2                      |
    |                 PADDLE_CURRENT_ENDPOINT                173.15.88.4:6070               |
    |                     PADDLE_TRAINERS_NUM                        4                      |
    |                PADDLE_TRAINER_ENDPOINTS  ... .5:6071,173.15.88.4:6070,173.15.88.4:6071|
    |                     PADDLE_RANK_IN_NODE                        0                      |
    |                 PADDLE_LOCAL_DEVICE_IDS                        0                      |
    |                 PADDLE_WORLD_DEVICE_IDS                     0,1,0,1                   |
    |                     FLAGS_selected_gpus                        0                      |
    |             FLAGS_selected_accelerators                        0                      |
    +=======================================================================================+

INFO 2022-04-03 16:58:50,321 launch_utils.py:514] details abouts PADDLE_TRAINER_ENDPOINTS can be found in log/endpoints.log, and detail running logs maybe found in log/workerlog.0
launch proc_id:226 idx:0
launch proc_id:229 idx:1
W0403 16:58:55.397342   342 gen_comm_id_helper.cc:120] connect addr=173.15.88.4:6070 failed 1 times with reason: Connection refused retry after 0.5 seconds
W0403 16:58:55.913834   342 gen_comm_id_helper.cc:120] connect addr=173.15.88.4:6070 failed 2 times with reason: Connection refused retry after 1 seconds
W0403 16:58:56.914870   342 gen_comm_id_helper.cc:120] connect addr=173.15.88.4:6070 failed 3 times with reason: Connection refused retry after 1.5 seconds
W0403 16:58:58.419081   342 gen_comm_id_helper.cc:120] connect addr=173.15.88.4:6070 failed 4 times with reason: Connection refused retry after 2 seconds
W0403 16:59:00.419550   342 gen_comm_id_helper.cc:120] connect addr=173.15.88.4:6070 failed 5 times with reason: Connection refused retry after 2.5 seconds
I0403 16:59:00.498201   226 gen_comm_id_helper.cc:181] Server listening on: 173.15.88.4:6070 successful.
I0403 16:59:02.923162   342 nccl_context.cc:74] init nccl context nranks: 4 local rank: 0 gpu id: 0 ring id: 0
I0403 16:59:02.922478   226 nccl_context.cc:74] init nccl context nranks: 4 local rank: 2 gpu id: 0 ring id: 0
W0403 16:59:03.412863   226 device_context.cc:404] Please NOTE: device: 0, GPU Compute Capability: 90.0, Driver API Version: 321.0, Runtime API Version: 3.1
W0403 16:59:03.413836   342 device_context.cc:404] Please NOTE: device: 0, GPU Compute Capability: 90.0, Driver API Version: 321.0, Runtime API Version: 3.1
W0403 16:59:03.421044   226 device_context.cc:417] device: 0, MIOpen Version: 2.11.0
W0403 16:59:03.431119   342 device_context.cc:417] device: 0, MIOpen Version: 2.11.0
I0403 16:59:03.452879   342 nccl_context.cc:107] init nccl context nranks: 4 local rank: 0 gpu id: 0 ring id: 10
I0403 16:59:03.452358   226 nccl_context.cc:107] init nccl context nranks: 4 local rank: 2 gpu id: 0 ring id: 10
/public/home/ac48p2il5w/anaconda3/envs/paddle_wjc_task/lib/python3.7/site-packages/paddle/tensor/creation.py:125: DeprecationWarning: `np.object` is a deprecated alias for the builtin `object`. To silence this warning, use `object` by itself. Doing this will not modify any behavior and is safe.
Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
  if data.dtype == np.object:
/public/home/ac48p2il5w/anaconda3/envs/paddle_wjc_task/lib/python3.7/site-packages/paddle/tensor/creation.py:125: DeprecationWarning: `np.object` is a deprecated alias for the builtin `object`. To silence this warning, use `object` by itself. Doing this will not modify any behavior and is safe.
Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
  if data.dtype == np.object:
2022-04-03 16:59:03,590-INFO: [topology.py:152:__init__] HybridParallelInfo: rank_id: 0, dp_degree: 4, mp_degree: 1, pp_degree: 1, dp_group: [0, 1, 2, 3], mp_group: [0], pp_group: [0], check/clip group: [0]
2022-04-03 16:59:03,590-INFO: [topology.py:152:__init__] HybridParallelInfo: rank_id: 2, dp_degree: 4, mp_degree: 1, pp_degree: 1, dp_group: [0, 1, 2, 3], mp_group: [2], pp_group: [2], check/clip group: [2]
Epoch  1 / 30
Epoch  1 / 30
/public/home/ac48p2il5w/anaconda3/envs/paddle_wjc_task/lib/python3.7/site-packages/paddle/nn/layer/norm.py:641: UserWarning: When training, we now always track global mean and variance.
  "When training, we now always track global mean and variance.")
/public/home/ac48p2il5w/anaconda3/envs/paddle_wjc_task/lib/python3.7/site-packages/paddle/nn/layer/norm.py:641: UserWarning: When training, we now always track global mean and variance.
  "When training, we now always track global mean and variance.")
step 0 / 14 - loss: 0.85710step 0 / 14 - loss: 0.84205

step 10 / 14 - loss: 0.05992step 10 / 14 - loss: 0.05753

Eval begin...
Eval begin...
step 0 / 4 - loss: 0.01912
step 0 / 4 - loss: 0.01973
Eval samples:  428
Epoch  2 / 30
Eval samples:  428
Epoch  2 / 30
step 0 / 14 - loss: 0.04044step 0 / 14 - loss: 0.03697

step 10 / 14 - loss: 0.02107
step 10 / 14 - loss: 0.02008
Eval begin...
Eval begin...
step 0 / 4 - loss: 0.00549
step 0 / 4 - loss: 0.00503
Eval samples:  428
Epoch  3 / 30
Eval samples:  428
Epoch  3 / 30
step 0 / 14 - loss: 0.01858step 0 / 14 - loss: 0.02017

step 10 / 14 - loss: 0.01667
step 10 / 14 - loss: 0.01475
Eval begin...
Eval begin...
step 0 / 4 - loss: 0.00300
step 0 / 4 - loss: 0.00350
Eval samples:  428
Epoch  4 / 30
Eval samples:  428
Epoch  4 / 30
step 0 / 14 - loss: 0.01376step 0 / 14 - loss: 0.01363

step 10 / 14 - loss: 0.01104
step 10 / 14 - loss: 0.01042
Eval begin...
Eval begin...
step 0 / 4 - loss: 0.00240
step 0 / 4 - loss: 0.00249
Eval samples:  428
Epoch  5 / 30
Eval samples:  428
Epoch  5 / 30
step 0 / 14 - loss: 0.01193
step 0 / 14 - loss: 0.01152
step 10 / 14 - loss: 0.01055
step 10 / 14 - loss: 0.01039
Eval begin...Eval begin...

step 0 / 4 - loss: 0.00166
step 0 / 4 - loss: 0.00154
Eval samples:  428
Epoch  6 / 30
Eval samples:  428
Epoch  6 / 30
step 0 / 14 - loss: 0.01019step 0 / 14 - loss: 0.00879

step 10 / 14 - loss: 0.00636
step 10 / 14 - loss: 0.00875
Eval begin...
Eval begin...
step 0 / 4 - loss: 0.00257
step 0 / 4 - loss: 0.00269
Eval samples:  428
Epoch  7 / 30
Eval samples:  428
Epoch  7 / 30
step 0 / 14 - loss: 0.00884
step 0 / 14 - loss: 0.00832
step 10 / 14 - loss: 0.01013step 10 / 14 - loss: 0.00839

Eval begin...
Eval begin...
step 0 / 4 - loss: 0.00227
step 0 / 4 - loss: 0.00179
Eval samples:  428
Epoch  8 / 30
Eval samples:  428
Epoch  8 / 30
step 0 / 14 - loss: 0.00957step 0 / 14 - loss: 0.00895

step 10 / 14 - loss: 0.00707step 10 / 14 - loss: 0.00848

Eval begin...Eval begin...

step 0 / 4 - loss: 0.00154
step 0 / 4 - loss: 0.00165
Eval samples:  428
Epoch  9 / 30
Eval samples:  428
Epoch  9 / 30
step 0 / 14 - loss: 0.00606
step 0 / 14 - loss: 0.00653
step 10 / 14 - loss: 0.00564
step 10 / 14 - loss: 0.00885
Eval begin...
Eval begin...
step 0 / 4 - loss: 0.00180
step 0 / 4 - loss: 0.00198
Eval samples:  428
Epoch  10 / 30
Eval samples:  428
Epoch  10 / 30
step 0 / 14 - loss: 0.00697step 0 / 14 - loss: 0.00953

step 10 / 14 - loss: 0.00577
step 10 / 14 - loss: 0.00713
Eval begin...
Eval begin...
step 0 / 4 - loss: 0.00132
step 0 / 4 - loss: 0.00098
Eval samples:  428
Epoch  11 / 30
Eval samples:  428
Epoch  11 / 30
step 0 / 14 - loss: 0.00611
step 0 / 14 - loss: 0.00576
step 10 / 14 - loss: 0.00705step 10 / 14 - loss: 0.00557

Eval begin...Eval begin...

step 0 / 4 - loss: 0.00229
step 0 / 4 - loss: 0.00296
Eval samples:  428
Epoch  12 / 30
Eval samples:  428
Epoch  12 / 30
step 0 / 14 - loss: 0.00789step 0 / 14 - loss: 0.00599

step 10 / 14 - loss: 0.00526
step 10 / 14 - loss: 0.00603
Eval begin...
Eval begin...
step 0 / 4 - loss: 0.00172
step 0 / 4 - loss: 0.00214
Eval samples:  428
Epoch  13 / 30
Eval samples:  428
Epoch  13 / 30
step 0 / 14 - loss: 0.00730
step 0 / 14 - loss: 0.00736
step 10 / 14 - loss: 0.00490step 10 / 14 - loss: 0.00524

Eval begin...
Eval begin...
step 0 / 4 - loss: 0.00129step 0 / 4 - loss: 0.00103

Eval samples:  428
Epoch  14 / 30
Eval samples:  428
Epoch  14 / 30
step 0 / 14 - loss: 0.00460step 0 / 14 - loss: 0.00474

step 10 / 14 - loss: 0.01072step 10 / 14 - loss: 0.01057

Eval begin...Eval begin...

step 0 / 4 - loss: 0.00576
step 0 / 4 - loss: 0.00567
Eval samples:  428
Epoch  15 / 30
Eval samples:  428
Epoch  15 / 30
step 0 / 14 - loss: 0.00579
step 0 / 14 - loss: 0.00819
step 10 / 14 - loss: 0.00540step 10 / 14 - loss: 0.00567

Eval begin...
Eval begin...
step 0 / 4 - loss: 0.00103
step 0 / 4 - loss: 0.00086
Eval samples:  428
Epoch  16 / 30
Eval samples:  428
Epoch  16 / 30
step 0 / 14 - loss: 0.00486
step 0 / 14 - loss: 0.00365
step 10 / 14 - loss: 0.00576step 10 / 14 - loss: 0.00696

Eval begin...
Eval begin...
step 0 / 4 - loss: 0.00199
step 0 / 4 - loss: 0.00195
Eval samples:  428
Epoch Eval samples:   17428
/Epoch   3017
 / 30
step 0 / 14 - loss: 0.00727
step 0 / 14 - loss: 0.00616
step 10 / 14 - loss: 0.00570step 10 / 14 - loss: 0.00670

Eval begin...
Eval begin...
step 0 / 4 - loss: 0.00239
step 0 / 4 - loss: 0.00206
Eval samples:  428
Epoch  18 / 30
Eval samples:  428
Epoch  18 / 30
step 0 / 14 - loss: 0.00725step 0 / 14 - loss: 0.00614

step 10 / 14 - loss: 0.00709
step 10 / 14 - loss: 0.00581
Eval begin...
Eval begin...
step 0 / 4 - loss: 0.00101
step 0 / 4 - loss: 0.00106
Eval samples: Eval samples:   428428

Epoch Epoch   1919  //  3030

step 0 / 14 - loss: 0.00431
step 0 / 14 - loss: 0.00360
step 10 / 14 - loss: 0.00403
step 10 / 14 - loss: 0.00461
Eval begin...
Eval begin...
step 0 / 4 - loss: 0.00208
step 0 / 4 - loss: 0.00230
Eval samples:  428
Epoch  20 / 30
Eval samples:  428
Epoch  20 / 30
step 0 / 14 - loss: 0.00542step 0 / 14 - loss: 0.00424

step 10 / 14 - loss: 0.00426
step 10 / 14 - loss: 0.00520
Eval begin...
Eval begin...
step 0 / 4 - loss: 0.00098
step 0 / 4 - loss: 0.00088
Eval samples:  428
Epoch  21 / 30
Eval samples:  428
Epoch  21 / 30
step 0 / 14 - loss: 0.00452
step 0 / 14 - loss: 0.00440
step 10 / 14 - loss: 0.00489step 10 / 14 - loss: 0.00504

Eval begin...
Eval begin...
step 0 / 4 - loss: 0.00619
step 0 / 4 - loss: 0.00630
Eval samples:  428
Epoch  22 / 30
Eval samples:  428
Epoch  22 / 30
step 0 / 14 - loss: 0.00808
step 0 / 14 - loss: 0.00936
step 10 / 14 - loss: 0.00833step 10 / 14 - loss: 0.00734

Eval begin...
Eval begin...
step 0 / 4 - loss: 0.00144
step 0 / 4 - loss: 0.00146
Eval samples:  428
Epoch  23 / 30
Eval samples:  428
Epoch  23 / 30
step 0 / 14 - loss: 0.00359
step 0 / 14 - loss: 0.00316
step 10 / 14 - loss: 0.00343step 10 / 14 - loss: 0.00360

Eval begin...Eval begin...

step 0 / 4 - loss: 0.00134
step 0 / 4 - loss: 0.00156
Eval samples:  428
Epoch  24 / 30
Eval samples:  428
Epoch  24 / 30
step 0 / 14 - loss: 0.00492
step 0 / 14 - loss: 0.00398
step 10 / 14 - loss: 0.00439step 10 / 14 - loss: 0.00321

Eval begin...
Eval begin...
step 0 / 4 - loss: 0.00180
step 0 / 4 - loss: 0.00185
Eval samples:  428
Epoch  25 / 30
Eval samples:  428
Epoch  25 / 30
step 0 / 14 - loss: 0.00396step 0 / 14 - loss: 0.00419

step 10 / 14 - loss: 0.00630step 10 / 14 - loss: 0.00398

Eval begin...Eval begin...

step 0 / 4 - loss: 0.00308step 0 / 4 - loss: 0.00300

Eval samples:  428
Epoch  26 / 30
Eval samples:  428
Epoch  26 / 30
step 0 / 14 - loss: 0.00535
step 0 / 14 - loss: 0.00487
step 10 / 14 - loss: 0.00413step 10 / 14 - loss: 0.00418

Eval begin...Eval begin...

step 0 / 4 - loss: 0.00072
step 0 / 4 - loss: 0.00091
Eval samples:  428
Epoch  27 / 30
Eval samples:  428
Epoch  27 / 30
step 0 / 14 - loss: 0.00315step 0 / 14 - loss: 0.00386

step 10 / 14 - loss: 0.00435step 10 / 14 - loss: 0.00286

Eval begin...
Eval begin...
step 0 / 4 - loss: 0.00116
step 0 / 4 - loss: 0.00125
Eval samples:  428
Epoch  28 / 30
Eval samples:  428
Epoch  28 / 30
step 0 / 14 - loss: 0.00396step 0 / 14 - loss: 0.00281

step 10 / 14 - loss: 0.00449step 10 / 14 - loss: 0.00416

Eval begin...
Eval begin...
step 0 / 4 - loss: 0.00104
step 0 / 4 - loss: 0.00141
Eval samples:  428
Epoch  29 / 30
Eval samples:  428
Epoch  29 / 30
step 0 / 14 - loss: 0.00542step 0 / 14 - loss: 0.00310

step 10 / 14 - loss: 0.00570
step 10 / 14 - loss: 0.00425
Eval begin...
Eval begin...
step 0 / 4 - loss: 0.00171
step 0 / 4 - loss: 0.00193
Eval samples:  428
Epoch  30 / 30
Eval samples:  428
Epoch  30 / 30
step 0 / 14 - loss: 0.00446step 0 / 14 - loss: 0.00518

step 10 / 14 - loss: 0.00387step 10 / 14 - loss: 0.00282

Eval begin...
Eval begin...
step 0 / 4 - loss: 0.00094
step 0 / 4 - loss: 0.00079
Eval samples:  428
Eval samples:  428
```

## 3.3 评估表格
|     |    |      |          |      |
| :------------ | ---------- | --------------- | ------ | ------|
| 序号         | 核心步骤    |  完成情况（成功/不成功） | 遇到问题 |解决方法（无法解决请注明）|
| 1   |  导入分布式训练所需要的依赖包                    | 完成 | 无 | 无 |
| 2 |  初始化分布式环境   | 完成 | PaddlePaddle 安装有时候会有一些问题、NCCL 初始化有问题![图片](https://user-images.githubusercontent.com/35827074/165877509-b84f5846-b175-4ab9-8ae3-eef66ed09047.png) | 使用 export 设置一些安装的库的环境变量，上述问题是 rocm 版本问题，需要使用 rocm-4.0.1 版本。 修改 rocm 版本的方法为. module switch compiler/rocm/4.0.1，再就是导入超算上的一些环境变量 export NCCL_IB_HCA=mlx5_0 export NCCL_SOCKET_IFNAME=eno1 export NCCL_IB_DISABLE=0 |
| 3 | 设置分布式训练需要的优化器                    | 完成 | 无 | 无 |
| 4 | 数据集拆分                     | 完成 | 示例里面没有数据集的拆分案例，不会使用数据集的拆分；使用 DistributedBatchSampler 采样器之后 DataLoader 中无法指定 batchsize 以及 shuffle 参数 | 分析 paddle 的分布式 API 底层以及结合其他深度学习框架分析，发现了 DistributedBatchSampler API，然后分析其底层实现，发现可以应用；分析 DataLoader 底层的源码，发现在指定 batch_sampler 参数之后不能指定 batchsize、shuffle 以及 drop_last 参数，然后在 DistributedBatchSampler 构建采样器的过程中指定。分布式数据集拆分使用 DistributedBatchSampler，通过使用 DistributedBatchSampler 构建一个分布式的采样器，其会将数据平均划分到多个设备中，然后将其输入到 Dataloader 函数中，参数为 batch_sampler，案例的全部代码已经在附录中给出。关于拆分部分如下：train_sampler = DistributedBatchSampler(train_dataset, 32, shuffle=True)   train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=2)   val_sampler = DistributedBatchSampler(val_dataset, 32)   val_loader = DataLoader(val_dataset, batch_sampler=val_sampler, num_workers=2) |
| 5 | 构建训练代码               |   完成  |    无      | 无 |
| 6 | 单机多卡分布式训练                   |  完成  |   在曙光超算上使用 SBATCH 作业提交方式时有环境的问题  |  申请 4 个 DCU，使用镜像的方式进行实现  |
| 7 | 多机多卡分布式训练                |  完成  |   无  |  注意再进行多机多卡时先要两个机器之间互相 ping 一下  |

* 总结：上述单机转为分布式的过程中，总体来说感觉还是可以的，动态图下 paddle 单机转为分布式的代码还是比较方便的，也有一些官网上的参考文档用于学习，但是有一些是在其参考文档中没有介绍的，例如数据集的拆分等这些需要自己去思考。

# 4、PyTorch 单机与分布式：
## 4.1、PyTorch 单机

PyTorch 单机下的流程和 Paddle 单机下的流程基本上是相似的，采用了和 Paddle 单机下相同的模型，相同的优化器与损失函数，过程如下：
- 1、导入相关库
- 2、构建数据集
- 3、定义模型
- 4、构建损失函数与优化器
- 5、训练模型
- 6、预测

## 4.2、PyTorch 分布式

PyTorch 单机转为分布式的具体流程如下：
- 1、导入分布式所需要的依赖包
```python
import torch.distributed as dist
```
- 2、初始化分布式环境，需要指定一下通信后端（我采用的是 NCCL），初始化方法（我采用的是 env 初始化），当前进程号以及总的进程数量。
```python
rank = int(os.environ["RANK"])
world_size = int(os.environ['WORLD_SIZE'])
gpu = int(os.environ['LOCAL_RANK'])
device = gpu
torch.cuda.set_device(gpu)
dist.init_process_group(backend="nccl", world_size=world_size, rank=rank)
```
3、构建模型，这里需要使用 torch.nn.parallel.DistributedDataParallel 构建其分布式的模型，这里没有采用 Dataparallel API，是因为其效率相对于 DistributedDataParallel 比较低。
```python
model = FaceNet(num_keypoints=15).to(device)
model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
```
4、构建优化器与损失函数
优化器采用 SGD，指定学习率与动量等参数。
```python
optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001, momentum=0.9)
```
损失函数采用均方误差损失函数。
```python
criterion = torch.nn.MSELoss()
```
5、数据集的拆分，首先使用 DistributedSampler 构建分布式数据集的拆分，这里可以指定一下是否需要进行 shuffle 以及 drop_last 等参数，然后使用 DataLoader 进行封装。
```python
train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
train_loader = torch.utils.data.DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size)
val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
val_loader = torch.utils.data.DataLoader(val_dataset, sampler=val_sampler, batch_size=batch_size)
```
6、构建训练代码，这里需要使用 train_sampler.set_epoch(epoch)设置一下，因为在 train_sampler 中使用了 shuffle 这个参数，而看其源码可以发现其依靠 self.epoch 这个参数进行随机种子的设置，所以需要在每个 epoch 训练时设置一下采样器的 self.epoch 这个参数，即通过 train_sampler.set_epoch(epoch)进行设置。训练代码如下，每一个 epoch 训练完进行 val 的验证评估：
```python
for epoch in range(total_epoch):
    model.train()
    train_sampler.set_epoch(epoch)
    for batch_id, data in enumerate(train_loader):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        if batch_id % 10 == 0:
            print("step %d / %d - loss: %.5f" % (batch_id, len(train_loader), loss.item()))
    model.eval()
    print("Eval begin...")
    for batch_id, data in enumerate(val_loader):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        if batch_id % 10 == 0:
            print("step %d / %d - loss: %.5f" % (batch_id, len(val_loader), loss.item()))
    print("Eval samples: ", len(val_dataset))
```
7、启动分布式任务
PyTorch 分布式下两种启动方式，我选择的是和 paddle 类似的一种方式即通过 torch.distributed.launch 进行启动，启动方式如下：
```bash
python -m torch.distributed.launch --nproc_per_node=4 train_multi.py
```
# 4、对比分析

下面对 PaddlePaddle 与 PyTorch 单机转为分布式进行对比性分析：
## 相似点：
PaddlePaddle 与 PyTorch 单机转为分布式的流程基本上是相似的，基本上遵循如下流程：导入分布式相关的库、初始化分布式环境、构建分布式的模型、构建优化器与损失函数、同时进行分布式数据集的拆分，最后构建训练代码，其整个流程都比较相似，相对来说 PaddlePaddle 与 PyTorch 单机转为分布式都是比较方便的。
## 不同点：
- 1、PaddlePaddle 内部有许多封装好的类，例如 paddle.Model 类，其内部封装了好多函数，例如 train_batch/fit 等函数，还加入了一些回调函数例如 EarlyStopping 等，可以比较方便地进行训练、测试的过程，比较容易使用。
- 2、对于单机转为分布式的过程，如果对数据集进行 shuffle 打乱时，PyTorch 需要在每个 epoch 训练开始时调用 train_sampler.set_epoch 函数即设置一下 shuffle 打乱的种子，但是 PaddlePaddle 如果对数据集进行 shuffle 打乱时，可以选择并不需要设置，因为其内部在每次打乱时会将 self.epoch 进行加一的操作，即自动改变了其数据打乱的种子，使用起来更加方便。
- 3、从使用方面来说，PaddlePaddle 的分布式初始化有时候会报错有时候能使用，其环境用起来感觉不太稳定，PyTorch 的分布式使用起来相对比较稳定，其初始化环境等功能实现都比较稳定。
- 4、从官方文档来说，PaddlePaddle 的分布式示例文档中感觉不太完善，例如 DistributedSampler 等的 API 没有在分布式示例文档中展现，paddle.Model 等 API 没有找到相关 API 文档的介绍；PyTorch 的分布式示例文档相对来说比较完善，包括其示例以及 API 的使用以及分布式通信的相关 API 都有其文档介绍。

# 5、附录
## 单机示例转为分布式的代码
```python
import numpy as np
import pandas as pd

import paddle
from paddle.io import Dataset
from paddle.vision.transforms import transforms
from paddle.vision.models import resnet18
# 导入必要分布式训练的依赖包
from paddle.distributed import fleet
from paddle.io import Dataset, BatchSampler, DataLoader, DistributedBatchSampler

# 数据文件的路径
# training.csv: 包含了用于训练的人脸关键点坐标和图像。
# test.csv: 包含了用于测试的人脸关键点图像, 没有标注关键点坐标。
# IdLookupTable.csv: 测试集关键点的位置的对应名称。
Train_Dir = './data/data60/training.csv'
Test_Dir = './data/data60/test.csv'
lookid_dir = './data/data60/IdLookupTable.csv'


class ImgTransforms:
    """
    图像预处理工具，用于将图像进行升维(96, 96) => (96, 96, 3)，
    并对图像的维度进行转换从 HWC 变为 CHW
    """

    def __init__(self, fmt):
        self.format = fmt

    def __call__(self, img):
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=2)
        img = img.transpose(self.format)

        if img.shape[0] == 1:
            img = np.repeat(img, 3, axis=0)
        return img


class FaceDataset(Dataset):
    def __init__(self, data_path, mode='train', val_split=0.2):
        self.mode = mode
        assert self.mode in ['train', 'val', 'test'], \
            "mode should be 'train' or 'test', but got {}".format(self.mode)
        # 读取数据的路径
        self.data_source = pd.read_csv(data_path)
        # 清洗数据, 数据集中有很多样本只标注了部分关键点, 这里有两种策略
        # 第一种, 将未标注的位置从上一个样本对应的关键点复制过来
        # self.data_source.fillna(method = 'ffill',inplace = True)
        # 第二种, 将包含有未标注的样本从数据集中移除
        self.data_source.dropna(how="any", inplace=True)
        self.data_label_all = self.data_source.drop('Image', axis=1)

        # 划分训练集和验证集合
        if self.mode in ['train', 'val']:
            np.random.seed(43)
            data_len = len(self.data_source)
            # 随机划分
            shuffled_indices = np.random.permutation(data_len)
            # 顺序划分
            # shuffled_indices = np.arange(data_len)
            self.shuffled_indices = shuffled_indices
            val_set_size = int(data_len * val_split)
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
        img = np.array(img, dtype='float32').reshape(96, 96)
        # 图像预处理操作
        img = self.transforms(img)
        label = np.array(self.data_label.iloc[idx, :], dtype='float32') / 96
        return img, label

    # 返回整个数据集的总数
    def __len__(self):
        return len(self.data_img)

# 模型的定义
# 对应 30 维度
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


def main():
    # 初始化 Fleet 环境
    fleet.init(is_collective=True)
    # 训练数据集和验证数据集
    train_dataset = FaceDataset(Train_Dir, mode='train')
    val_dataset = FaceDataset(Train_Dir, mode='val')
    # 测试数据集
    test_dataset = FaceDataset(Test_Dir, mode='test')
    # 初始化模型
    model = FaceNet(num_keypoints=15)
    # 优化器的设置
    optim = paddle.optimizer.Adam(learning_rate=1e-3, parameters=model.parameters())
    # 构建分布式优化器
    optim = fleet.distributed_optimizer(optim)
    # 通过 Fleet API 获取分布式 model，用于支持分布式训练
    model = fleet.distributed_model(model)
    # 数据集的拆分  构建分布式数据集
    train_sampler = DistributedBatchSampler(train_dataset, 32, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=2)
    val_sampler = DistributedBatchSampler(val_dataset, 32)
    val_loader = DataLoader(val_dataset, batch_sampler=val_sampler, num_workers=2)
    epoch = 30
    for eop in range(epoch):
        # train_sampler.set_epoch(eop)
        # 设置为训练模式
        model.train()
        print("Epoch ", eop + 1, "/", epoch)
        for batch_id, data in enumerate(train_loader()):
            img, label = data
            label.stop_gradient = True
            # 前向传播
            out = model(img)
            # 均方损失函数
            loss = paddle.nn.functional.mse_loss(input=out, label=label)
            loss_data = loss.numpy()
            # 反向传播
            loss.backward()
            optim.step()
            model.clear_gradients()
            if batch_id % 10 == 0:
                print("step %d / %d - loss: %.5f" % (batch_id, len(train_loader), loss_data))
        model.eval()
        print("Eval begin...")
        for batch_id, data in enumerate(val_loader()):
            img, label = data
            label.stop_gradient = True
            # 前向传播
            out = model(img)
            # 均方损失函数
            loss = paddle.nn.functional.mse_loss(input=out, label=label)
            loss_data = loss.numpy()
            if batch_id % 10 == 0:
                print("step %d / %d - loss: %.5f" % (batch_id, len(val_loader), loss_data))
        print("Eval samples: ", len(val_dataset))


if __name__ == "__main__":
    main()
```
