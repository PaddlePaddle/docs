# 迁移经验汇总

这里提供 CV 各个方向从 PyTorch 迁移到飞桨的基本流程、常用工具、定位问题的思路及解决方法。

## 一、迁移概述

模型迁移本质上需要从模型训练与预测的角度去完成该任务，保证训练结果与预测结果与参考代码保持一致。

* 在模型预测过程中，模型组网、数据处理与加载、评估指标等均需要严格对齐，否则无法产出与参考代码完全相同的模型及预测结果。
* 在训练阶段，除了模型结构等元素，训练的损失函数、梯度、训练超参数、初始化方法以及训练精度也是我们需要迁移并对齐的内容。一个完整的模型训练包括定义模型结构并初始化，将处理后的数据送进网络，对输出的内容与真值计算损失函数，并反向传播与迭代的过程。

### 1.1 迁移流程

本章节从模型训练和推理需要的基本操作出发，对迁移工作进行任务分解，如下图所示。同时对每个环节进行对齐验证，检查每个环节飞桨和 PyTorch 模型在同样输入下的输出是否一致，以便快速发现问题，降低问题定位的难度。

![procedure](../../images/procedure.png)

1. **迁移准备**：迁移工作首先需要安装必要的软件和工具（包括飞桨、PyTorch 或 TensorFlow 的安装、差异核验工具的安装等），然后准备要迁移的模型以及使用的数据集，同时了解源代码结构、跑通模型训练，最后对源代码进行解析，统计缺失算子。
2. **模型前向对齐**：这是迁移工作最基本的部分。在搭建神经网络时，会使用到框架提供的 API（内置的模块、函数等）。在迁移时，需要对这些 API 转换成飞桨对应的 API。转换前后的模型应具有相同的网络结构，使用相同的模型权重时，对于相同的输入，二者的输出结果应该一致。有时同一个神经网络有不同的版本、同一个版本有不同的实现方式或者在相同的神经网络下使用不同的超参，这些差别会对最终的收敛精度和性能造成一定影响。通常，我们以神经网络作者本身的实现为准，也可以参考不同框架（例如飞桨、TensorFlow、PyTorch 等）的官方实现或其他主流开源工具箱（例如 MMDetection）。PyTorch 的大部分 API 在飞桨中可找到对应 API，可以直接对模型组网部分代码涉及的 API 进行手动转换。为了判断转换后的飞桨模型组网能获得和 PyTorch 参考实现同样的输出，可将两个模型参数固定，并输入相同伪数据，观察两者的产出差异是否在阈值内。
3. **小数据集数据读取对齐**：数据读取对齐为了验证数据加载、数据预处理、数据增强与原始代码一致。为了快速验证数据读取对齐，建议准备一个小数据集（训练集和验证集各 8~16 张图像即可，压缩后数据大小建议在 20MB 以内）。
4. **评估指标对齐**：评估指标是模型精度的度量。在计算机视觉中，不同任务使用的评估指标有所不同（比如，在图像分类任务中，常用的指标是 Top-1 准确率与 Top-5 准确率；在图像语义分割任务中，常用的指标是 mIOU）为了检验迁移后的模型能否达到原模型的精度指标，需要保证使用的评估指标与原始代码一致以便对照。飞桨提供了一系列 Metric 计算类，而 PyTorch 中目前可以通过组合的方式实现或者调用第三方的 API。
5. **损失函数对齐**：损失函数是训练模型时的优化目标，使用的损失函数会影响模型的精度。在模型迁移时，需要保证迁移后模型训练时使用的损失函数与原始代码中使用的损失函数一致，以便二者对照。飞桨与 PyTorch 均提供了常用的损失函数。
6. **模型训练超参对齐**：模型的训练超参包括学习率、优化器、正则化策略等。这些超参数指定了模型训练过程中网络参数的更新方式，训练超参数的设置会影响到模型的收敛速度及收敛精度。同样地，在模型迁移时，需要保证迁移前后模型使用的训练超参数一致，以便对照二者的收敛情况。飞桨中的优化器有 `paddle.optimizer` 等一系列实现，PyTorch 中则有 `torch.optim` 等一系列实现。完成超参对齐后，可以使用反向梯度对齐统一验证该模块的正确性。
7. **反向梯度对齐**：在完成前向对齐的基础上，还需进行反向梯度对齐。反向梯度对齐的目的是确保迁移后的模型反向传播以及权重更新的行为与原始模型一致，同时也是对上一步**模型训练超参对齐**的验证。具体的检验方法是通过两次（或以上）迭代训练进行检查，若迁移前后的模型第二轮训练的 loss 一致，则可以认为二者反向已对齐。
8. **训练集数据读取对齐**：相同的神经网络使用不同的数据训练和测试得到的结果往往会存在差异。因此，为了能复现原始代码的精度，需要保证使用的数据完全相同，包括数据集的版本、使用的数据预处理方法和流程、使用的数据增强方式等。
9. **网络初始化对齐**：对于不同的深度学习框架，网络初始化在大多情况下，即使值的分布完全一致，也无法保证值完全一致，这也是模型迁移不确定性比较大的地方。CNN 对于模型初始化相对不敏感，在迭代轮数与数据集足够的情况下，最终精度指标基本接近。而 transformer 系列模型、超分模型、领域自适应算法对于初始化比较敏感，需要对初始化进行重点检查。如果十分怀疑初始化导致的问题，建议将参考的初始化权重转成飞桨模型权重，加载该初始化模型训练，检查收敛精度。
10. **训练精度对齐**：模型训练的最终结果是为了得到一个精度达标的模型。不同的框架版本、是否为分布式训练等可能会对训练精度有影响，在迁移前需要分析清楚对标的框架、硬件等信息。对比迁移前后模型的训练精度，若二者的差值在可以接受的误差范围内，则精度对齐完成。同时，如果在相同的硬件条件下，迁移前后的模型训练速度应接近。若二者差异非常大，则需要排查原因。
11. **模型预测验证**：模型训练完成之后，需要使用测试集对该模型基于训练引擎进行预测，确认预测结果与实际一致。
其中，2~5 是迁移的重点，其他模块比如反向梯度、优化器、学习率生成等，要么本身结构单一，要么依赖已开发完成的网络结果才能和对标脚本形成对比，这些模块的脚本开发难度较小。

**【注意事项】**

如果遇到迁移时间较长的项目，建议：

* 根据自己的时间、资源、战略部署评估是否进行此项目迁移。
* 在决定迁移的情况下，参照本迁移指南中的对齐操作对模型、数据、优化方式等对齐，以最快的时间排除问题。
* 模型的实现具有相通性，为提升模型迁移效率，可参考和借鉴已实现模型的代码。飞桨提供了大规模的官方模型库，包含经过产业实践长期打磨的主流模型以及在国际竞赛中的夺冠模型，算法总数超过 500 多个，详细请参考链接：https://www.paddlepaddle.org.cn/modelbase。

**【获取更多飞桨信息】**

可以通过[API 文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/index_cn.html)了解飞桨各接口的相关信息；还可通过[教程](https://aistudio.baidu.com/aistudio/course/introduce/1297)系统掌握如何使用飞桨进行训练、调试、调优、推理。

### 1.2 准备工作

在开始模型迁移前，开发者需要**准备一个小型的数据集**，以便快速验证。在下载好原始参考代码之后，首先需要**运行参考代码**，确保能正常运行并达到预期的精度；**分析参考代码的项目结构**，并以此作为参考**构建飞桨项目**。最后，还需要检查原始代码中使用到的 API 在飞桨中是否均有提供。若飞桨未提供相应的 API，需要以组合实现、自定义算子等方式实现相应的功能。

#### 1.2.1 环境准备

在开始模型迁移前，开发者需要**准备好飞桨与 PyTorch 的环境**，并安装**差异核验工具**，以便后续检查迁移后的模型与原始模型是否对齐。

##### 1.2.1.1 安装飞桨

推荐使用飞桨最新版本。如果不确定自己安装的是否是最新版本，可以进入[网站](https://www.paddlepaddle.org.cn/whl/linux/gpu/develop.html)下载对应的包并查看时间戳。更多版本或者环境下的安装可以参考：[飞桨安装指南](https://www.paddlepaddle.org.cn/)

```bash
# 安装 GPU 版本的飞桨
pip install paddlepaddle-gpu
# 安装 CPU 版本的 Paddle
pip install paddlepaddle
```



运行 python，输入下面的命令：

```python
import paddle
paddle.utils.run_check()
```



如果输出下面的内容，则说明飞桨安装成功。

```python
PaddlePaddle is installed successfully! Let's start deep learning with PaddlePaddle now.
```



**【FAQ】**

1. 如何安装飞桨的 develop 版本？

    在飞桨修复了框架的问题或者新增了 API 和功能之后，若需要立即使用，可以采用以下方式安装最新的 develop 版本：

    进入[飞桨官网](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/develop/install/pip/linux-pip.html)，选择 develop 版本，并根据自己的情况选择其他字段，根据生成的安装信息安装，当选择 Linux-pip-CUDA10.2 字段后，就可以按照下面的信息安装。

    ```shell
    python -m pip install paddlepaddle-gpu==0.0.0.post102 -f https://www.paddlepaddle.org.cn/whl/linux/gpu/develop.html
    ```

####

在对齐验证的流程中，我们依靠 reprod_log 差异核验工具查看飞桨和 PyTorch 同样输入下的输出是否相同。这样的查看方式具有标准统一、比较过程方便等优势。

Reprod_log 是一个用于 numpy 数据记录和对比工具，通过传入需要对比的两个 numpy 数组就可以在指定的规则下得到数据之差是否满足期望的结论。其主要接口的说明可以查看其 [GitHub 主页](https://github.com/PaddlePaddle/models/tree/release/2.3/tutorials/reprod_log)。

安装 reprod_log 的命令如下：

```bash
pip3 install reprod_log --force-reinstall
```

##### 1.2.1.2 安装 PyTorch

对于 PyTorch 的安装，请参阅 [PyTorch 官网](https://pytorch.org/get-started/locally/)，选择操作系统和 CUDA 版本，使用相应的命令安装。

运行 Python，输入以下命令，如果可以正常输出，则说明 PyTorch 安装成功。

```python
import torch
print(torch.__version__)
# 如果安装的是 cpu 版本，可以按照下面的命令确认 torch 是否安装成功
# 期望输出为 tensor([1.])
print(torch.Tensor([1.0]))
# 如果安装的是 gpu 版本，可以按照下面的命令确认 torch 是否安装成功
# 期望输出为 tensor([1.], device='cuda:0')
print(torch.Tensor([1.0]).cuda())
```



##### 1.2.1.3 安装 reprod_log

为了减少数据对比中标准不一致、人工对比过程繁杂的问题，我们提供了数据对比日志工具 reprod_log。reprod_log 是一个用于 numpy 数据记录和对比工具，通过传入需要对比的两个 numpy 数组就可以在指定的规则下得到数据之差是否满足期望的结论。

reprod_log 可作为辅助自查和验收工具应用到论文复现赛、模型迁移等场景。针对迁移场景，可以在对齐验证的流程中通过 reprod_log 查看飞桨和 PyTorch 同样输入下的输出是否相同。此查看方式具有标准统一，比较过程方便等优势。

reprod_log 主要功能如下：

- 存取指定节点的输入输出 tensor
- 基于文件的 tensor 读写
- 2 个字典的对比验证
- 对比结果的输出与记录

详细介绍与使用方法可以参考 GitHub ：https://github.com/PaddlePaddle/models/blob/release%2F2.3/tutorials/reprod_log/README.md

**【reprod_log 使用 demo】**

下面基于前面的 MobileNetV3 示例代码，给出如何使用该工具。

文件夹中包含 write_log.py 和 check_log_diff.py 文件，其中 write_log.py 中给出了 ReprodLogger 类的使用方法，check_log_diff.py 给出了 ReprodDiffHelper 类的使用方法，依次运行两个 Python 文件，使用下面的方式运行代码。

```python
进入文件夹
cd pipeline/reprod_log_demo
随机生成矩阵，写入文件中
python3 write_log.py
进行文件对比，输出日志
python3 check_log_diff.py
```



最终会输出以下内容：

```bash
2021-09-28 01:07:44,832 - reprod_log.utils - INFO - demo_test_1:
2021-09-28 01:07:44,832 - reprod_log.utils - INFO -     mean diff: check passed: True, value: 0.0
2021-09-28 01:07:44,832 - reprod_log.utils - INFO - demo_test_2:
2021-09-28 01:07:44,832 - reprod_log.utils - INFO -     mean diff: check passed: False, value: 0.3336232304573059
2021-09-28 01:07:44,832 - reprod_log.utils - INFO - diff check failed
```



可以看出：对于 key 为 demo_test_1 的矩阵，由于 diff 为 0，小于设置的阈值 1e-6，核验成功；对于 key 为 demo_test_2 的矩阵，由于 diff 为 0.33，大于设置的阈值 1e-6，核验失败。

**【reprod_log 在迁移中的应用】**

针对迁移场景，基于 reprod_log 的结果记录模块，产出下面若干文件：

```bash
result
├── log
├── data_paddle.npy
├── data_ref.npy
├── forward_paddle.npy
├── forward_ref.npy           # 与 forward_paddle.npy 作为一并核查的文件对
├── metric_paddle.npy
├── metric_ref.npy            # 与 metric_paddle.npy 作为一并核查的文件对
├── loss_paddle.npy
├── loss_ref.npy              # 与 loss_paddle.npy 作为一并核查的文件对
├── losses_paddle.npy
├── losses_ref.npy            # 与 losses_paddle.npy 作为一并核查的文件对
```

基于 reprod_log 的 ReprodDiffHelper 模块，产出下面 5 个日志文件。

```bash
log
├── data_diff.log            # data_paddle.npy 与 data_torch.npy 生成的 diff 结果文件
├── forward_diff.log         # forward_paddle.npy 与 forward_torch.npy 生成的 diff 结果文件
├── metric_diff.log          # metric_paddle.npy 与 metric_torch.npy 生成的 diff 结果文件
├── loss_diff.log            # loss_paddle.npy 与 loss_torch.npy 生成的 diff 结果文件
├── backward_diff.log        # losses_paddle.npy 与 losses_torch.npy 生成的 diff 结果文件
```

上述文件的生成代码都需要开发者进行开发。



#### 1.2.2 准备数据和模型

- 了解待迁移模型的输入输出格式。
- 准备好全量数据（训练集/验证集/测试集），用于后续的全量数据对齐训练、评估和预测。
- 准备伪输入数据（fake input data）以及伪标签（fake label），与模型输入 shape、type 等保持一致，用于后续模型前向对齐。
  - 在模型前向对齐过程中，不需要考虑数据集模块等其他模块，此时使用伪数据是将模型结构和数据部分解耦非常合适的一种方式。
  - 将伪数据以文件的形式存储下来，也可以保证飞桨与参考代码的模型结构输入是完全一致的，更便于排查问题。
  - 伪数据可以通过如下代码生成，data 目录下也提供了生成好的伪数据（./data/fake_*.npy）。以 [MobileNetV3](https://github.com/PaddlePaddle/models/blob/release%2F2.2/tutorials/mobilenetv3_prod/Step1-5/utilities.py) 为例，通过运行生成伪数据的脚本可以参考：[mobilenetv3_prod/Step1-5/utilities.py](https://github.com/PaddlePaddle/models/blob/release%2F2.2/tutorials/mobilenetv3_prod/Step1-5/utilities.py)。

```python
defgen_fake_data():
    fake_data=np.random.rand(1, 3, 224, 224).astype(np.float32) - 0.5
    fake_label=np.arange(1).astype(np.int64)
    np.save("fake_data.npy", fake_data)
    np.save("fake_label.npy", fake_label)
```

#### 1.2.3 分析并运行参考代码

需在特定设备(CPU/GPU)上，利用少量伪数据，跑通参考代码的预测过程(前向)以及至少 2 轮(iteration)迭代过程，用于生成和迁移代码进行对比的结果。

对于复杂的神经网络，完整的训练需要耗时几天甚至几个月，如果仅以最终的训练精度和结果做参考，会极大地降低开发效率。因此，我们可以利用少量数据进行少量迭代训练缩短时间（该迭代是执行了数据预处理、权重初始化、正向计算、loss 计算、反向梯度计算和优化器更新之后的结果，覆盖了网络训练的全部环节），并以此为对照展开后续的开发工作。

以 MobileNetV3 的复现为例，PyTorch 项目如下：

https://github.com/PaddlePaddle/models/tree/release/2.3/tutorials/mobilenetv3_prod/Step1-5/mobilenetv3_ref

其目录结构如下：

```bash
torchvision
    |--datasets                             # 数据集加载与处理相关代码
    |--models                               # 模型组网相关代码
         |--__init__.py
         |--_utils.py                       # 工具类及函数
         |--misc_torch.py                   # 组成网络的子模块
         |--mobilenet_v3_torch.py           # 模型网络结构定义
    |--transforms                           # 数据增强实现
    |--__init__.py
    |--_internally_replaced_utils.py
__init__.py
metric.py                                   # 评价指标实现
presets.py                                  # 数据增强设定
train.py                                    # 模型训练代码
utils.py                                    # 工具类及函数
```

为便于对比，已将使用到的 torchvision 库的代码提取出来放到 torchvision 目录下，三个子目录内的代码分别对应 torchvision.Datasets, torchvision.models 和 torchvision.transforms。checkpoint.pth 是保存的权重文件，以便前向对齐时加载使用。

为了便于实例的演示，可将以下参考代码下载到本地。

```bash
# 克隆参考代码所在的项目 repo 到本地
git clone https://github.com/PaddlePaddle/models.git
cd model/tutorials/mobilenetv3_prod/
```



#### 1.2.4 构建迁移项目

为了便于对比，建议按照 PyTorch 项目的结构构建飞桨项目。例如，对于上述的 MobileNetV3 项目的复现，按照原始项目的结构，可构建飞桨项目如下：

```bash
paddlevision
    |--datasets                             # 数据集加载与处理相关代码
    |--models                               # 模型组网相关代码
         |--__init__.py
         |--_utils.py                       # 工具类及函数
         |--misc_paddle.py                  # 组成网络的子模块
         |--mobilenet_v3_paddle.py          # 模型网络结构定义
    |--transforms                           # 数据增强实现
    |--__init__.py
__init__.py
metric.py                                   # 评价指标实现
presets.py                                  # 数据增强设定
train.py                                    # 模型训练代码
utils.py                                    # 工具类及函数
```

#### 1.2.5 评估算子和 API

使用飞桨搭建神经网络流程与 PyTorch 类似，但支持的算子存在差异，需要在进行模型迁移前找出飞桨缺失的算子。

飞桨算子由各种 Python/C++ API 组成，包括基础数据结构、数据处理、模型组网、 优化器、分布式等。下面以其中较为常见的 3 类 API 算子进行介绍。

- 数据框架算子，包括张量处理、设备、基本数据类型等，如`paddle.abs`、`paddle.int64`、paddle.CPUPlace 等。
- 数据处理算子，包括数据集定义与加载、数据采用、多进程数据读取器算子，如`paddle.io.Dataset`、`paddle.io.BatchSampler`、`paddle.io.DataLoader`等。
- 模型组网算子，包括网络构建中使用到的卷积、全连接、激活等算子，如`paddle.nn.Conv2D`、`paddle.nn.Linear、``paddle.nn.ReLU`等。

更多关于飞桨 API 的介绍，请参考：[飞桨 API 文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/Overview_cn.html)。

**【统计缺失算子】**

统计缺失算子时，在代码库找到网络结构及实现训练功能的 Python 文件（名称一般为 train.py model.py 等等），在脚本文件中查找所有相关算子（含数据框架类、数据预处理、网络结构算子），通过 AI 映射表，查找对应的飞桨算子。例如`torch.nn.ReLU`对应飞桨算子为`paddle.nn.ReLU。`飞桨与 PyTorch 关于 API 的映射关系可以参考：[API 映射表](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/model_convert/pytorch_api_mapping_cn.html)。

若该网页未能找到对应的算子，则可继续在[飞桨 API](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/index_cn.html)中搜索算子名称。

如果依然没有对应的飞桨算子，则计入缺失，可以尝试通过其他 API 算子组合实现、自定义算子实现或者在[Paddle repo](https://github.com/PaddlePaddle/Paddle/issues/new/choose)中提出一个新的需求。

**【注意事项】**

针对相同功能的算子，飞桨的命名可能与 PyTorch 不同；同名算子参数与功能也可能与 PyTorch 有区别。如 [paddle.optimizer.lr.StepDecay](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/optimizer/lr/StepDecay_cn.html#stepdecay)与[torch.optim.lr_scheduler.StepLR](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.StepLR.html#torch.optim.lr_scheduler.StepLR) 。

**【缺失算子处理策略】**

如果遇到飞桨不包含的算子或者 API，例如：某些算法实现存在调用了外部算子，而且飞桨也不包含该算子实现；或者其他框架存在的 API 或者算子，但是飞桨中没有这些算子。

1. 尝试使用替代实现进行迁移，比如下面的 PyTorch 代码，飞桨中可以通过 slice + concat API 的组合的形式进行功能实现。

    ```python
    torch.stack([
                    per_locations[:, 0] - per_box_regression[:, 0],
                    per_locations[:, 1] - per_box_regression[:, 1],
                    per_locations[:, 0] + per_box_regression[:, 2],
                    per_locations[:, 1] + per_box_regression[:, 3],
                ], dim=1)
    ```

2. 尝试使用飞桨的自定义算子功能：参考[自定义算子](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/custom_op/index_cn.html#zidingyisuanzi)。此方式存在一定的代码开发量。
3. 考虑通过自定义算子方式，使用其他已有第三方算子库实现：参考[PaddleDetection 自定义算子编译文档](https://github.com/PaddlePaddle/PaddleDetection/blob/release%2F2.5/ppdet/ext_op/README.md)。
4. 如果缺失的功能和算子无法规避，或者组合算子性能较差，严重影响网络的训练和推理，欢迎给飞桨提[ISSUE](https://github.com/PaddlePaddle/Paddle/issues/new/choose)，列出飞桨不支持的实现，飞桨开发人员会根据优先级进行实现。

## 二、模型前向对齐

模型前向对齐是模型迁移的最核心部分。在这一部分工作中，需要保证迁移之后的模型结构与原始模型一致，当两个模型固定为相同的权重时，对于相同的输入，二者的输出结果相同（即“前向对齐”）。模型前向对齐主要包括以下步骤：

1. **网络结构代码转换**：将定义模型网络结构的 PyTorch 代码转换成飞桨代码。由于飞桨定义模型网络结构的方式与 PyTorch 非常相似，因此主要的工作在于 API 的转换（例如内置的模块、函数）。
2. **权重转换**：只有当两个模型的权重相同时，对于相同的输入数据，两个模型才能输出相同的结果（即“前向对齐”）。为了检查模型前向对齐，需要使飞桨模型具有与原始模型相同的权重，因此需要把原始模型的权重转换为飞桨格式的模型权重，供飞桨模型加载。
3. **模型前向对齐验证**：让转换前后的模型加载相同的权重，给两个模型输入相同的数据，利用差异核验工具检查两个模型的输出是否一致。若二者的差异小于一定的阈值，表明迁移后的模型实现了前向对齐验证。

### 2.1 网络结构代码转换

PyTorch 中的网络模块继承`torch.nn.Module`，而飞桨的网络模块则继承`paddle.nn.Layer`。二者对网络结构的定义方式相似，都是在`__init__`中定义构成网络的各个子模块，并实现`forward`函数，定义网络的前向传播方式。因此本步骤的工作主要在于将原始代码调用的 PyTorch API（即原始代码中 import 的 torch 包的类、函数，例如 `torch.nn` 中的模块及 `torch.nn.functional` 中的函数等）替换成相应的飞桨 API。需要注意的是，飞桨 API 与 PyTorch 中对应的 API 在功能与参数上存在一定区别，转换时需要留意。更多详细内容请参见“[解读网络结构转换](https://ku.baidu-int.com/knowledge/HFVrC7hq1Q/yKeL8Lljko/QuZY-m-iji/_U88vYhMWELV8m)”。

**【基本流程】**

由于 PyTorch 的 API 和飞桨的 API 非常相似，可以参考 [PyTorch-飞桨 API 映射表](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/08_api_mapping/pytorch_api_mapping_cn.html)，组网部分代码直接进行手动转换即可。

以最简单的卷积神经网络 LeNet 为例，PyTorch 代码如下：

```python
import torch
from torch import nn

class LeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()
        self.num_classes = num_classes
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(400, 120),
            nn.Linear(120, 84), nn.Linear(84, num_classes))

    def forward(self, inputs):
        x = self.features(inputs)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
```

根据源代码，该神经网络由卷积层`Conv2d`、最大池化层`MaxPool2d`、`ReLu`激活函数及全连接层`Linear`组成。这里使用到的 PyTorch API 与对应的飞桨 API 如下表所示：

| PyTorch             | 飞桨                 |
| ------------------- | -------------------- |
| torch.nn.Module     | paddle.nn.Layer      |
| torch.nn.Conv2d     | paddle.nn.Conv2D     |
| torch.nn.ReLu       | paddle.nn.ReLu       |
| torch.nn.MaxPool2d  | paddle.nn.MaxPool2D  |
| torch.nn.Sequential | paddle.nn.Sequential |
| torch.nn.Linear     | paddle.nn.Linear     |
| torch.flatten       | paddle.flatten       |

根据上述对应，直接进行转换，得到飞桨代码：

```python
import paddle
from paddle import nn

class LeNet(nn.Layer):
    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()
        self.num_classes = num_classes
        self.features = nn.Sequential(
            nn.Conv2D(1, 6, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2D(2, 2),
            nn.Conv2D(6, 16, 5, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2D(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(400, 120),
            nn.Linear(120, 84), nn.Linear(84, num_classes))

    def forward(self, inputs):
        x = self.features(inputs)
        x = paddle.flatten(x, 1)
        x = self.fc(x)
        return x
```

**【实战】**

MobilnetV3 网络结构的 PyTorch 实现: [mobilenetv3_prod/Step1-5/mobilenetv3_ref/torchvision/models/mobilenet_v3_torch.py](https://github.com/PaddlePaddle/models/blob/release%2F2.2/tutorials/mobilenetv3_prod/Step1-5/mobilenetv3_ref/torchvision/models/mobilenet_v3_torch.py)

对应转换后的飞桨实现: [mobilenetv3_prod/Step1-5/mobilenetv3_paddle/paddlevision/models/mobilenet_v3_paddle.py](https://github.com/PaddlePaddle/models/blob/release%2F2.2/tutorials/mobilenetv3_prod/Step1-5/mobilenetv3_paddle/paddlevision/models/mobilenet_v3_paddle.py)

**【FAQ】**

1. 有什么其他没有在映射表中的 PyTorch API 是可以用飞桨中 API 实现的呢？

    有，例如：

    - `torch.masked_fill`函数的功能目前可以使用`paddle.where`进行实现，可以参考[链接](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/faq/train_cn.html#paddletorch-masked-fillapi)。
    - `pack_padded_sequence`和`pad_packed_sequence`这两个 API 目前飞桨中没有实现，可以直接在 RNN 或者 LSTM 的输入中传入`sequence_length`来实现等价的功能（可参考[ISSUE](https://github.com/PaddlePaddle/Paddle/issues/36882)）。

2. 为什么`nn.AvgPool2D` 会存在不能对齐的问题？

    [paddle.nn.AvgPool2D](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/AvgPool2D_cn.html#avgpool2d)需要将 `exclusive` 参数设为 `False` ，结果才能 PyTorch 的默认行为一致。

3. 权重初始化，飞桨与 PyTorch 差异有哪些？

    合适的初始化方法能够帮助模型快速地收敛或者达到更高的精度。对于权重初始化，飞桨提供了大量的初始化方法，包括 `Constant`, `KaimingUniform`, `KaimingNormal`,  `TruncatedNormal`,  `Uniform`,  `XavierNormal`,  `XavierUniform`等，然而由于不同框架的差异性，部分 API 中参数提供的默认初始化方法有区别。

    在 PyTorch 中，通常调用 `torch.nn.init`下的函数来定义权重初始化方式，例如：

    ```python
    for m in self.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out")
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.zeros_(m.bias)
    ```

    飞桨提供的初始化方式为在定义模块时直接修改 API 的 ParamAttr 属性，与 torch.nn.init 等系列 API 的使用方式不同，例如：

    ```python
    self.conv = nn.Conv2D(in_channels, out_channels, kernel_size, weight_attr=nn.initializer.KaimingNormal())
    self.fc = paddle.nn.Linear(4096, 512,
                weight_attr=init.Normal(0, 0.01),
                bias_attr=init.Constant(0))
    ```

    另外，PaddleDetection 中实现了与`torch.nn.init`系列 API 完全对齐的初始化 API，包括`uniform_`, `normal_`, `constant_`, `ones_`, `zeros_`, `xavier_uniform_`, `xavier_normal_`, `kaiming_uniform_`, `kaiming_normal_`, `linear_init_`, `conv_init_`，可以参考[initializer.py](https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/initializer.py)，查看更多的实现细节。

    权重初始化更多详细介绍请参阅 [模型参数初始化对齐方法](https://github.com/PaddlePaddle/models/blob/release/2.3/tutorials/article-implementation/initializer.md)。有关初始化 API 的详细介绍，请参阅[初始化 API 官网文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/Overview_cn.html#chushihuaxiangguan)。

### 2.2 权重转换

只有当两个模型的权重相同时，对于相同的输入数据，两个模型才能输出相同的结果（即“前向对齐”）。Paddle 框架保存的模型文件名一般后缀为 `'.pdparams'` ，PyTorch 框架保存的模型文件名一般后缀为 `'.pt'`、`'.pth'` 或者 `'.bin'` 。因此需要把原始模型的权重转换为飞桨格式的模型权重，供飞桨模型加载。更多详细内容请参加“[模型权重转换详解](https://ku.baidu-int.com/knowledge/HFVrC7hq1Q/yKeL8Lljko/QuZY-m-iji/1x5E1jKmodoa3F)”。

**【基本流程】**

组网代码转换完成之后，需要对模型权重进行转换。权重的转换分为两个步骤：

1. **获取 PyTorch 权重**：下载 PyTorch 的模型权重，或随机生成并保存 PyTorch 权重
2. **转换为飞桨权重**：对二者存在差异的参数进行转换，并保存为飞桨的权重字典

下面以 MobileNetV3_small 为例介绍转换的过程：

如果 PyTorch repo 中已经提供权重，那么可以直接下载并进行后续的转换。如果没有提供，则可以基于 PyTorch 代码，随机生成一个初始化权重(定义完 model 以后，使用`torch.save()` API 保存模型权重)：

```python
from torchvision.models import mobilenet_v3_small

model = mobilenet_v3_small()
torch.save(model.state_dict(), "mobilenet_v3_small.pth")
```



然后将生成的 PyTorch 权重 (mobilenet_v3_small.pth) 转换为飞桨模型权重，转换代码如下（代码解释详见代码后的 FAQ）：

```python
import numpy as np
import torch
import paddle

def torch2paddle():
    torch_path = "./data/mobilenet_v3_small.pth"
    paddle_path = "./data/mv3_small_paddle.pdparams"
    torch_state_dict = torch.load(torch_path)
    fc_names = ["classifier"]
    paddle_state_dict = {}
    for k in torch_state_dict:
        if "num_batches_tracked" in k:  # 飞桨中无此参数，无需保存
            continue
        v = torch_state_dict[k].detach().cpu().numpy()
        flag = [i in k for i in fc_names]
        if any(flag) and "weight" in k:
            new_shape = [1, 0] + list(range(2, v.ndim))
            print(
                f"name: {k}, ori shape: {v.shape}, new shape: {v.transpose(new_shape).shape}"
            )
            v = v.transpose(new_shape)   # 转置 Linear 层的 weight 参数
        # 将 torch.nn.BatchNorm2d 的参数名称改成 paddle.nn.BatchNorm2D 对应的参数名称
        k = k.replace("running_var", "_variance")
        k = k.replace("running_mean", "_mean")
        # 添加到飞桨权重字典中
        paddle_state_dict[k] = v
    paddle.save(paddle_state_dict, paddle_path)
```



一般地，对于一个需要迁移的模型，可以用查看其权重字典的键：

```python
>>> from torchvision.models import mobilenet_v3_small
>>> model = mobilenet_v3_small()
>>> torch_dict = model.state_dict()
>>> torch_dict.keys()
odict_keys(['features.0.0.weight', 'features.0.1.weight', 'features.0.1.bias', 'features.0.1.running_mean', 'features.0.1.running_var', 'features.0.1.num_batches_tracked', 'features.1.block.0.0.weight', 'features.1.block.0.1.weight', 'features.1.block.0.1.bias', 'features.1.block.0.1.running_mean', 'features.1.block.0.1.running_var', 'features.1.block.0.1.num_batches_tracked', 'features.1.block.1.fc1.weight', 'features.1.block.1.fc1.bias', 'features.1.block.1.fc2.weight', 'features.1.block.1.fc2.bias', 'features.1.block.2.0.weight', 'features.1.block.2.1.weight', 'features.1.block.2.1.bias', 'features.1.block.2.1.running_mean', 'features.1.block.2.1.running_var', 'features.1.block.2.1.num_batches_tracked', 'features.2.block.0.0.weight', 'features.2.block.0.1.weight', 'features.2.block.0.1.bias', 'features.2.block.0.1.running_mean', 'features.2.block.0.1.running_var', 'features.2.block.0.1.num_batches_tracked', 'features.2.block.1.0.weight', 'features.2.block.1.1.weight', 'features.2.block.1.1.bias', 'features.2.block.1.1.running_mean', 'features.2.block.1.1.running_var', 'features.2.block.1.1.num_batches_tracked', 'features.2.block.2.0.weight', 'features.2.block.2.1.weight', 'features.2.block.2.1.bias', 'features.2.block.2.1.running_mean', 'features.2.block.2.1.running_var', 'features.2.block.2.1.num_batches_tracked', 'features.3.block.0.0.weight', 'features.3.block.0.1.weight', 'features.3.block.0.1.bias', 'features.3.block.0.1.running_mean', 'features.3.block.0.1.running_var', 'features.3.block.0.1.num_batches_tracked', 'features.3.block.1.0.weight', 'features.3.block.1.1.weight', 'features.3.block.1.1.bias', 'features.3.block.1.1.running_mean', 'features.3.block.1.1.running_var', 'features.3.block.1.1.num_batches_tracked', 'features.3.block.2.0.weight', 'features.3.block.2.1.weight', 'features.3.block.2.1.bias', 'features.3.block.2.1.running_mean', 'features.3.block.2.1.running_var', 'features.3.block.2.1.num_batches_tracked', 'features.4.block.0.0.weight', 'features.4.block.0.1.weight', 'features.4.block.0.1.bias', 'features.4.block.0.1.running_mean', 'features.4.block.0.1.running_var', 'features.4.block.0.1.num_batches_tracked', 'features.4.block.1.0.weight', 'features.4.block.1.1.weight', 'features.4.block.1.1.bias', 'features.4.block.1.1.running_mean', 'features.4.block.1.1.running_var', 'features.4.block.1.1.num_batches_tracked', 'features.4.block.2.fc1.weight', 'features.4.block.2.fc1.bias', 'features.4.block.2.fc2.weight', 'features.4.block.2.fc2.bias', 'features.4.block.3.0.weight', 'features.4.block.3.1.weight', 'features.4.block.3.1.bias', 'features.4.block.3.1.running_mean', 'features.4.block.3.1.running_var', 'features.4.block.3.1.num_batches_tracked', 'features.5.block.0.0.weight', 'features.5.block.0.1.weight', 'features.5.block.0.1.bias', 'features.5.block.0.1.running_mean', 'features.5.block.0.1.running_var', 'features.5.block.0.1.num_batches_tracked', 'features.5.block.1.0.weight', 'features.5.block.1.1.weight', 'features.5.block.1.1.bias', 'features.5.block.1.1.running_mean', 'features.5.block.1.1.running_var', 'features.5.block.1.1.num_batches_tracked', 'features.5.block.2.fc1.weight', 'features.5.block.2.fc1.bias', 'features.5.block.2.fc2.weight', 'features.5.block.2.fc2.bias', 'features.5.block.3.0.weight', 'features.5.block.3.1.weight', 'features.5.block.3.1.bias', 'features.5.block.3.1.running_mean', 'features.5.block.3.1.running_var', 'features.5.block.3.1.num_batches_tracked', 'features.6.block.0.0.weight', 'features.6.block.0.1.weight', 'features.6.block.0.1.bias', 'features.6.block.0.1.running_mean', 'features.6.block.0.1.running_var', 'features.6.block.0.1.num_batches_tracked', 'features.6.block.1.0.weight', 'features.6.block.1.1.weight', 'features.6.block.1.1.bias', 'features.6.block.1.1.running_mean', 'features.6.block.1.1.running_var', 'features.6.block.1.1.num_batches_tracked', 'features.6.block.2.fc1.weight', 'features.6.block.2.fc1.bias', 'features.6.block.2.fc2.weight', 'features.6.block.2.fc2.bias', 'features.6.block.3.0.weight', 'features.6.block.3.1.weight', 'features.6.block.3.1.bias', 'features.6.block.3.1.running_mean', 'features.6.block.3.1.running_var', 'features.6.block.3.1.num_batches_tracked', 'features.7.block.0.0.weight', 'features.7.block.0.1.weight', 'features.7.block.0.1.bias', 'features.7.block.0.1.running_mean', 'features.7.block.0.1.running_var', 'features.7.block.0.1.num_batches_tracked', 'features.7.block.1.0.weight', 'features.7.block.1.1.weight', 'features.7.block.1.1.bias', 'features.7.block.1.1.running_mean', 'features.7.block.1.1.running_var', 'features.7.block.1.1.num_batches_tracked', 'features.7.block.2.fc1.weight', 'features.7.block.2.fc1.bias', 'features.7.block.2.fc2.weight', 'features.7.block.2.fc2.bias', 'features.7.block.3.0.weight', 'features.7.block.3.1.weight', 'features.7.block.3.1.bias', 'features.7.block.3.1.running_mean', 'features.7.block.3.1.running_var', 'features.7.block.3.1.num_batches_tracked', 'features.8.block.0.0.weight', 'features.8.block.0.1.weight', 'features.8.block.0.1.bias', 'features.8.block.0.1.running_mean', 'features.8.block.0.1.running_var', 'features.8.block.0.1.num_batches_tracked', 'features.8.block.1.0.weight', 'features.8.block.1.1.weight', 'features.8.block.1.1.bias', 'features.8.block.1.1.running_mean', 'features.8.block.1.1.running_var', 'features.8.block.1.1.num_batches_tracked', 'features.8.block.2.fc1.weight', 'features.8.block.2.fc1.bias', 'features.8.block.2.fc2.weight', 'features.8.block.2.fc2.bias', 'features.8.block.3.0.weight', 'features.8.block.3.1.weight', 'features.8.block.3.1.bias', 'features.8.block.3.1.running_mean', 'features.8.block.3.1.running_var', 'features.8.block.3.1.num_batches_tracked', 'features.9.block.0.0.weight', 'features.9.block.0.1.weight', 'features.9.block.0.1.bias', 'features.9.block.0.1.running_mean', 'features.9.block.0.1.running_var', 'features.9.block.0.1.num_batches_tracked', 'features.9.block.1.0.weight', 'features.9.block.1.1.weight', 'features.9.block.1.1.bias', 'features.9.block.1.1.running_mean', 'features.9.block.1.1.running_var', 'features.9.block.1.1.num_batches_tracked', 'features.9.block.2.fc1.weight', 'features.9.block.2.fc1.bias', 'features.9.block.2.fc2.weight', 'features.9.block.2.fc2.bias', 'features.9.block.3.0.weight', 'features.9.block.3.1.weight', 'features.9.block.3.1.bias', 'features.9.block.3.1.running_mean', 'features.9.block.3.1.running_var', 'features.9.block.3.1.num_batches_tracked', 'features.10.block.0.0.weight', 'features.10.block.0.1.weight', 'features.10.block.0.1.bias', 'features.10.block.0.1.running_mean', 'features.10.block.0.1.running_var', 'features.10.block.0.1.num_batches_tracked', 'features.10.block.1.0.weight', 'features.10.block.1.1.weight', 'features.10.block.1.1.bias', 'features.10.block.1.1.running_mean', 'features.10.block.1.1.running_var', 'features.10.block.1.1.num_batches_tracked', 'features.10.block.2.fc1.weight', 'features.10.block.2.fc1.bias', 'features.10.block.2.fc2.weight', 'features.10.block.2.fc2.bias', 'features.10.block.3.0.weight', 'features.10.block.3.1.weight', 'features.10.block.3.1.bias', 'features.10.block.3.1.running_mean', 'features.10.block.3.1.running_var', 'features.10.block.3.1.num_batches_tracked', 'features.11.block.0.0.weight', 'features.11.block.0.1.weight', 'features.11.block.0.1.bias', 'features.11.block.0.1.running_mean', 'features.11.block.0.1.running_var', 'features.11.block.0.1.num_batches_tracked', 'features.11.block.1.0.weight', 'features.11.block.1.1.weight', 'features.11.block.1.1.bias', 'features.11.block.1.1.running_mean', 'features.11.block.1.1.running_var', 'features.11.block.1.1.num_batches_tracked', 'features.11.block.2.fc1.weight', 'features.11.block.2.fc1.bias', 'features.11.block.2.fc2.weight', 'features.11.block.2.fc2.bias', 'features.11.block.3.0.weight', 'features.11.block.3.1.weight', 'features.11.block.3.1.bias', 'features.11.block.3.1.running_mean', 'features.11.block.3.1.running_var', 'features.11.block.3.1.num_batches_tracked', 'features.12.0.weight', 'features.12.1.weight', 'features.12.1.bias', 'features.12.1.running_mean', 'features.12.1.running_var', 'features.12.1.num_batches_tracked', 'classifier.0.weight', 'classifier.0.bias', 'classifier.3.weight', 'classifier.3.bias'])
```

对于大部分 API，飞桨的权重保存方式与 PyTorch 对应 API 的权重保存方式一致，但存在少数 API 存在区别。对于其他可能遇到的权重保存方式存在差异的 API，可通过以上方式打印出模型权重字典的全部键，再根据差异情况编写转换代码进行转换。

**【FAQ】**

权重转换过程中，PyTorch 和飞桨有哪些参数存在差异？

在权重转换的时候，需要注意`paddle.nn.Linear`以及`paddle.nn.BatchNorm2D`等 API 的权重保存格式和名称等与 PyTorch 稍有差异：

- `nn.Linear`层的 weight 参数：飞桨与 PyTorch 的参数存在互为转置的关系，因此在转换时需要进行转置，这可以参考上述的 torch2paddle 函数。有时还会遇到线性层被命名为 conv 的情况，但是我们依旧需要进行转置。
- `nn.BatchNorm2D`参数：这个 API 在飞桨中包含 4 个参数`weight`,`bias`,`_mean`,`_variance`，torch.nn.BatchNorm2d 包含 5 个参数`weight`,`bias`,`running_mean`,`running_var`,`num_batches_tracked`。 其中，`num_batches_tracked`在飞桨中没有用到，剩下 4 个的对应关系为
  - `weight`->`weight`
  - `bias`->`bias`
  - `_variance`->`running_var`
  - `_mean`->`running_mean`

在权重转换的时候，不能只关注参数的名称，比如有些`paddle.nn.Linear`层，定义的变量名称为`conv`，这种情况下需要进行权重转置。权重转换时，建议同时打印飞桨和 PyTorch 对应权重的`shape`，以防止名称相似但是 `shape`不同的参数权重转换报错。

### 2.3 模型前向对齐验证

本步骤的目的是验证前面的网络结构代码转换的正确性，检查迁移前后模型结构是否一致。因此，需要让转换前后的模型加载相同的权重，给两个模型输入相同的数据，检查两个模型的输出是否一致。若二者的差异小于一定的阈值，则迁移后的模型前向对齐得到了验证。

**【基本流程】**

1. **PyTorch 前向传播**：定义 PyTorch 模型，加载权重，固定 seed，基于 numpy 生成随机数，转换为 `torch.Tensor`，送入网络，获取输出。
2. **飞桨前向传播**：定义飞桨模型，加载步骤 2.2 中转换后的权重，固定 seed，基于 numpy 生成随机数，转换为 `paddle.Tensor`，送入网络，获取输出。
3. **比较输出差异**：比较两个输出 tensor 的差值，如果小于阈值，即可完成自测。

**【实战】**

Mobilenetv3 模型前向对齐验证可以参考[mobilenetv3_prod/Step1-5/01_test_forward.py](https://github.com/PaddlePaddle/models/blob/release%2F2.2/tutorials/mobilenetv3_prod/Step1-5/01_test_forward.py)。

**【核验】**

对于待迁移的项目，前向对齐核验流程如下。

1. 准备输入：伪数据

    - 方式 1：使用参考代码的 dataloader，生成一个 batch 的数据，保存下来，在前向对齐时，直接从文件中读入。
    - 方式 2：固定随机数种子，生成 numpy 随机矩阵，转化 tensor。

2. 获取并保存输出：

    飞桨/PyTorch：dict，其中 key 为 tensor 的 name（自定义），value 为 tensor 的值。最后将 dict 保存到文件中。建议命名为 `forward_paddle.npy` 和 `forward_pytorch.npy`。

3. 自测：使用 reprod_log 加载 2 个文件，使用 report 功能，记录结果到日志文件中，建议命名为 `forward_diff_log.txt`，观察 diff，二者 diff 小于特定的阈值即可。

4. 提交内容：新建文件夹，将 `forward_paddle.npy`、`forward_pytorch.npy` 与 `forward_diff_log.txt` 文件放在文件夹中，后续的输出结果和自查日志也放在该文件夹中，一并打包上传即可。

5. 注意：

    - 飞桨与 PyTorch 保存的 dict 的 key 需要保持相同，否则 report 过程可能会提示 key 无法对应，从而导致 report 失败，之后的 **【核验】** 环节也是如此。
    - 如果是固定随机数种子，建议将伪数据保存到 dict 中，方便 check 参考代码和飞桨的输入是否一致。

**【注意事项】**

- 模型在前向对齐验证时，需要调用`model.eval()`方法，保证组网中的随机量被关闭，比如 BatchNorm、Dropout 等。
- 给定相同的输入数据，为保证可复现性，如果有随机数生成，固定相关的随机种子。
- 网络结构对齐后，尽量使用训练好的预训练模型和真实的图片进行前向差值计算，使结果更准确。
- 在前向过程中，如果数据预处理过程运行出错，请先将 `paddle.io.DataLoader` 的 `num_workers` 参数设为 0，然后根据单个进程下的报错日志定位出具体的 bug。

**【FAQ】**

1. 在复杂的网络结构中，前向结果对不齐怎么办？

    输出 diff 可以使用`np.mean(np.abs(o1 - o2))`进行计算。一般如果 diff 小于 1e-6，可以认为前向已对齐。如果最终输出结果 diff 较大，可以用以下两种方法排查：

    - 可以按照模块排查问题，比如依次获取 backbone、neck、head 输出等，查看问题具体出现在哪个子模块，再进到子模块详细排查。
    - 如果最终输出结果差值较大，使用二分的方法进行排查，比如说 ResNet50，包含 1 个 stem、4 个 res-stage、global avg-pooling 以及最后的 fc 层，那么完成模型组网和权重转换之后，如果模型输出没有对齐，可以尝试输出中间某一个 res-stage 的 tensor 进行对比，如果相同，则向后进行排查；如果不同，则继续向前进行排查，以此类推，直到找到导致没有对齐的操作。

2. 飞桨已经有了对于经典模型结构的实现，我还要重新实现一遍么？

    建议自己根据 PyTorch 代码重新实现一遍，一方面是对整体的模型结构更加熟悉，另一方面也保证模型结构和权重完全对齐。

## 三、小数据集数据读取对齐

为了快速验证数据读取对齐以及后续的训练、评估、预测，建议准备一个小数据集（训练集和验证集各 8~16 张图像即可，压缩后数据大小建议在 20MB 以内），放在 lite_data 文件夹下。若使用 Imagenet，可以使用自带的 [lite_data.tar](https://github.com/PaddlePaddle/models/blob/release%2F2.2/tutorials/mobilenetv3_prod/Step6/test_images/lite_data.tar)，并解压于 lite_data 下。

对于一个数据集，一般有以下一些信息需要重点关注：

-  数据集名称、下载地址
- 训练集/验证集/测试集图像数量、类别数量、分辨率等
- 数据集标注格式、标注信息
- 数据集通用的预处理方法

**【基本流程】**

1. **Dataset 迁移**：使用 `paddle.io.Dataset` 完成数据集的单个样本读取。飞桨中数据集相关的 API 为 `paddle.io.Dataset`，PyTorch 中对应为 `torch.utils.data.Dataset`，二者功能一致，在绝大多数情况下，可以使用该类构建数据集。

    此接口是描述 Dataset 方法和行为的抽象类，在具体实现的时候，需要继承这个基类，实现其中的`__getitem__`和`__len__`方法。

    除了参考代码中相关实现，也可以参考待迁移项目中的说明。

2. **DataLoader 迁移**：迁移完 Dataset 之后，可以使用`paddle.io.DataLoader`，构建 Dataloader，对数据进行组 batch、批处理，送进网络进行计算。

    `paddle.io.DataLoader`可以进行数据加载，将数据分成批数据，并提供加载过程中的采样。PyTorch 对应的实现为`torch.utils.data.DataLoader`，二者在功能上一致，仅在参数方面稍有差异：

    - 飞桨增加了`use_shared_memory`参数来选择是否使用共享内存加速数据加载过程。

**【实战】**

Mobilenet v3 迁移使用方法可以参考[mobilenetv3_prod/Step1-5/README.md](https://github.com/PaddlePaddle/models/blob/release%2F2.2/tutorials/mobilenetv3_prod/Step1-5/README.md#4.2)。准备`ImageNet 小数据集`的脚本可以参考[prepare.py](https://github.com/littletomatodonkey/AlexNet-Prod/blob/tipc/pipeline/Step2/prepare.py)，数据预处理和 Dataset、Dataloader 的检查可以参考[mobilenetv3_prod/Step1-5/02_test_data.py](https://github.com/PaddlePaddle/models/blob/release%2F2.2/tutorials/mobilenetv3_prod/Step1-5/02_test_data.py)。

**【注意事项】**

- 如果代码中存在 random.random 或者其他 random 库的，可以先把随机数替换成固定数值，避免 random 对对齐结果产生影响。设置 seed 的方式当调用 random 库的顺序不同时产生的随机值也不同，可能导致对齐存在问题。
- 迁移项目一般会提供数据集的名称以及基本信息。下载完数据之后，建议先检查是否与迁移项目描述一致，否则可能存在的问题有：
  - 数据集年份不同，比如 PyTorch 项目使用了 MS-COCO2014 数据集，而下载的是 MS-COCO2017 数据集。如果不对其进行检查，可能会导致最终训练的数据量等与 PyTorch 项目有差异；
  - 数据集使用方式不同。一些项目可能只是抽取了该数据集的子集进行方法验证，此时需要注意抽取方法，需要保证抽取出的子集完全相同。
- 构建数据集时，会涉及到一些预处理方法。以 CV 领域为例，飞桨提供了一些现成的视觉类操作 API，具体可以参考：[paddle.vision 类 API](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/vision/Overview_cn.html)。对应地，PyTorch 中的数据处理 API 可以参考：[torchvision.transforms 类 API](https://pytorch.org/vision/stable/transforms.html)，大部分的实现均可以找到对应的飞桨实现。此外，

    - 有些自定义的数据处理方法，如果不涉及到深度学习框架的部分，可以直接复用。
    - 对于特定任务中的数据预处理方法，比如说图像分类、检测、分割等，如果没有现成的 API 可以调用，可以参考官方模型套件中的一些实现方法，比如 PaddleClas、PaddleDetection、PaddleSeg 等。

- 如果使用飞桨提供的数据集 API，比如 `paddle.vision.datasets.Cifar10` 等，可能无法完全与参考代码在数据顺序上保持一致。如果是全量数据使用，对结果不会有影响，使用时对数据预处理和后处理进行排查即可；如果是按照比例选取子集进行训练，则建议重新根据参考代码实现数据读取部分，保证子集完全一致。

**【FAQ】**

1. 如果使用飞桨提供的数据集 API 如 `paddle.vision.datasets.Cifar10`，不能实现数据增强完全对齐怎么办？

    这些数据集的实现都是经过广泛验证的，可以使用。因此只需要完成数据预处理和后处理进行排查就好。`数据集+数据处理`的部分可以通过评估指标对齐完成自查。

2. 还有其他导致不能对齐的因素么？

    - 预处理方法顺序不一致：预处理的方法相同，顺序不同，比如先 padding 再做归一化与先做归一化再 padding，得到的结果可能是不同的。
    - 没有关闭 shuffle：在评估指标对齐时，需要固定 batch size，关闭 Dataloader 的 shuffle 操作。

## 四、评估指标对齐

评估指标是模型精度的度量。在计算机视觉中，不同任务使用的评估指标有所不同（比如，在图像分类任务中，常用的指标是 Top-1 准确率与 Top-5 准确率；在图像语义分割任务中，常用的指标是 mIOU）为了检验迁移后的模型能否达到原模型的精度指标，需要保证使用的评估指标与原始代码一致以便对照。在这一步骤中，需要将原始代码中的评价指标替换成飞桨的等价实现。在前面的步骤中，已经检查了两个模型已前向对齐。因此，在此基础上让两个模型固定相同的权重，给两个模型输入相同的数据，分别计算评估指标，若二者评估指标结果相同，则说明评估指标对齐完成。

飞桨提供了一系列 Metric 计算类，比如说`Accuracy`, `Auc`, `Precision`, `Recall`等，而 PyTorch 中，目前可以通过组合的方式实现评估指标的计算，或者调用 [torchmetrics](https://torchmetrics.readthedocs.io/en/latest/)。在迁移过程中，需要注意保证对于该模块，给定相同的输入，二者输出完全一致。

**【基本流程】**

1. **获取 PyTorch 模型评估结果**：定义 PyTorch 模型，加载训练好的权重，获取评估结果。
2. **获取飞桨模型评估结果**：定义飞桨模型，加载训练好的权重（需要是从 PyTorch 转换得到），获取评估结果。
3. **比较结果差异**：使用 reprod_log 比较两个结果的差值，如果小于阈值，即可完成自测。

**【实战】**

评估指标对齐检查方法可以参考文档：[mobilenetv3_prod/Step1-5/README.md](https://github.com/PaddlePaddle/models/blob/release%2F2.2/tutorials/mobilenetv3_prod/Step1-5/README.md#4.3)

**【核验】**

对于待迁移的项目，评估指标对齐核验流程如下。

1. 输入：dataloader, model
2. 输出：

    - 飞桨/PyTorch：dict，key 为 tensor 的 name（自定义），value 为具体评估指标的值。最后将 dict 使用 reprod_log 保存到各自的文件中，建议命名为`metric_paddle.npy`和`metric_pytorch.npy`。
    - 自测：使用 reprod_log 加载 2 个文件，使用 report 功能，记录结果到日志文件中，建议命名为`metric_diff_log.txt`，观察 diff，二者 diff 小于特定的阈值即可。

3. 提交内容：将`metric_paddle.npy`、`metric_pytorch.npy`与`metric_diff_log.txt`文件备份到`result`和`result/log`新建的文件夹中，后续的输出结果和自查日志也对应放在文件夹中，一并打包上传即可。
4. 注意：

    - 这部分需要使用真实数据
    - 需要检查 PyTorch 项目是否只是抽取了验证集/测试集中的部分文件。这种情况下，需要保证飞桨和参考代码中 dataset 使用的数据集一致。

**【FAQ】**

有哪些会导致评估出现精度偏差呢？

- 使用 dataloader 参数没有保持一致：例如需要检查飞桨代码与 PyTorch 代码在 `paddle.io.DataLoader` 的 `drop_last` 参数上是否保持一致 (文档[链接](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/io/DataLoader_cn.html#dataloader)，如果不一致，最后不够 batch-size 的数据可能不会参与评估，导致评估结果有差异。
- 转换到不同设备上运行代码：在识别或者检索过程中，为了加速评估过程，往往会将评估函数由 CPU 实现改为 GPU 实现，由此会带来评估函数输出的不一致。这是由于 sort 函数对于相同值的排序结果不同带来的。在迁移的过程中，如果可以接受轻微的指标不稳定，可以使用飞桨的 sort 函数，如果对于指标非常敏感，同时对速度性能要求很高，可以给飞桨提[ISSUE](https://github.com/PaddlePaddle/Paddle/issues/new/choose)，由研发人员根据优先级开发。
- 评估参数和训练参数不一致：在检测任务中，评估流程往往和训练流程有一定差异，例如 RPN 阶段 NMS 的参数等。因此，需要仔细检查评估时的超参数，不要将训练超参和评估超参混淆。
- 评估时数据过滤规则不一致：在 OCR 等任务中，需要注意评估过程也会对 GT 信息进行修正，比如大小写等，也会过滤掉一些样本。这种情况下，需要注意过滤规则，确保有效评估数据集一致。

## 五、损失函数对齐

损失函数是模型训练时的优化目标，使用的损失函数会影响模型的精度。在模型迁移时，需要保证迁移后模型训练时使用的损失函数与原始代码中使用的损失函数一致，以便二者对照。在前面的步骤中，已经检查了两个模型已前向对齐。因此，在此基础上让两个模型固定相同的权重，给两个模型输入相同的数据，分别计算损失函数，若二者 loss 相同，则说明损失函数对齐完成。

飞桨与 PyTorch 均提供了很多损失函数，用于模型训练，具体的 API 映射表可以参考：[Loss 类 API 映射列表](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/model_convert/pytorch_api_mapping_cn.html#lossapi)。

**【基本流程】**

1. **计算 PyTorch 损失**：定义 PyTorch 模型，加载权重，加载 fake data 和 fake label（或者固定 seed，基于 numpy 生成随机数），转换为 `torch.Tensor`，送入网络，获取 loss 结果。
2. **计算飞桨损失**：定义飞桨模型，加载 fake data 和 fake label（或者固定 seed，基于 numpy 生成随机数），转换为 `paddle.Tensor`，送入网络，获取 loss 结果。
3. **比较结果差异**：使用 reprod_log 比较两个结果的差值，如果小于阈值，即可完成自测。

**【差异分析】**

以 CrossEntropyLoss 为例，主要区别为：

飞桨提供了对软标签、指定 softmax 计算维度的支持。如果 PyTorch 使用的 loss function 没有指定的 API，则可以尝试通过组合 API 的方式，实现自定义的 loss function。

**【实战】**

本部分可以参考[mobilenetv3_prod/Step1-5/04_test_loss.py](https://github.com/PaddlePaddle/models/blob/release%2F2.2/tutorials/mobilenetv3_prod/Step1-5/04_test_loss.py)。

**【核验】**

对于待迁移的项目，损失函数对齐核验流程如下。

1. 输入：fake data & label
2. 输出：飞桨/PyTorch：dict，key 为 tensor 的 name（自定义），value 为具体评估指标的值。最后将 dict 使用 reprod_log 保存到各自的文件中，建议命名为 `loss_paddle.npy` 和 `loss_pytorch.npy`。
3. 自测：使用 reprod_log 加载 2 个文件，使用 report 功能，记录结果到日志文件中，建议命名为`loss_diff_log.txt`，观察 diff，二者 diff 小于特定的阈值即可。

**【注意事项】**

- 计算 loss 的时候，建议设置`model.eval()`，避免模型中随机量的问题。
- 飞桨目前支持在 `optimizer` 中通过设置 `params_groups` 的方式设置不同参数的更新方式，可以参考[代码示例](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/optimizer/optimizer.py#L140) 。
- 某些模型训练时，会使用梯度累加策略，即累加到一定 step 数量之后才进行参数更新，这时在实现上需要注意对齐。

**【FAQ】**

1. 为什么`nn.CrossEntropyLoss`出现不能对齐问题？

    `paddle.nn.CrossEntropyLoss` 默认是在最后一维(axis=-1)计算损失函数，而 `torch.nn.CrossEntropyLoss` 是在 axis=1 的地方计算损失函数，因此如果输入的维度大于 2，需要保证计算的维(axis)相同，否则可能会出错。

## 六、模型训练超参对齐

模型的训练超参包括学习率、优化器、正则化策略等。这些超参数指定了模型训练过程中网络参数的更新方式，训练超参数的设置会影响到模型的收敛速度及收敛精度。在模型迁移时，需要保证迁移前后模型使用的训练超参数一致，以便对照二者的收敛情况。

**【基本流程】**

本部分对齐建议对照飞桨[优化器 API 文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/optimizer/Overview_cn.html)、[飞桨正则化 API 文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/regularizer/L2Decay_cn.html)与参考代码的优化器实现进行对齐，用之后的反向对齐统一验证该模块的正确性。

- **优化器**：飞桨中的 optimizer 有`paddle.optimizer`等一系列实现，PyTorch 中则有`torch.optim`等一系列实现。需要仔细对照飞桨与参考代码的优化器参数实现，确保优化器参数严格对齐。
- **学习率策略**：主要用于指定训练过程中的学习率变化曲线。可以将定义好的学习率策略，不断 step，得到每一步对应的学习率值。对于迁移代码和 PyTorch 代码，学习率在整个训练过程中在相同的轮数相同的 iter 下应该保持一致，可以通过打印学习率值或者可视化二者学习率的 log 来查看差值。可以参考学习率对齐代码 [Step1-5/05_test_backward.py#L23](https://github.com/PaddlePaddle/models/blob/release%2F2.2/tutorials/mobilenetv3_prod/Step1-5/05_test_backward.py#L23)。
- **正则化策略**：在模型训练中，L2 正则化可以防止模型对训练数据过拟合，L1 正则化可以用于得到稀疏化的权重矩阵。飞桨中有`paddle.regularizer.L1Decay`与`paddle.regularizer.L2Decay` API。PyTorch 中，torch.optim 集成的优化器只有 L2 正则化方法，直接在构建 optimizer 的时候，传入`weight_decay`参数即可。在如 Transformer 或者少部分 CNN 模型中，存在一些参数不做正则化(正则化系数为 0)的情况。这里需要找到这些参数并对齐取消实施正则化策略，可以参考[这里](https://github.com/PaddlePaddle/PaddleClas/blob/f677f61ea3b34281968d1059f4f8f13c1d787e67/ppcls/arch/backbone/model_zoo/resnest.py#L74)，对特定参数进行修改。

**【区别】**

以 SGD 等优化器为例，飞桨与 PyTorch 的优化器区别主要如下：

- 飞桨中，需要首先构建学习率策略，再传入优化器对象中；对于 PyTorch，如果希望使用更丰富的学习率策略，需要先构建优化器，再传入学习率策略类 API。
- 飞桨在优化器中增加了对梯度裁剪的支持，在训练 GAN、NLP、多模态等任务时较为常用。
- 飞桨的 SGD 不支持动量更新、动量衰减和 Nesterov 动量。若需实现这些功能，请使用`paddle.optimizer.Momentum` API。
- 飞桨的 optimizer 中支持 L1Decay/L2Decay。

**【FAQ】**

1. 怎么实现参数分组学习率策略？

    飞桨目前支持在 `optimizer` 中通过设置 `params_groups` 的方式设置不同参数的更新方式，可以参考[代码示例](https://github.com/PaddlePaddle/Paddle/blob/166ff39a20f39ed590a0cf868c2ad2f15cf0bbb1/python/paddle/optimizer/optimizer.py#L138) 。

2. 有没有什么其他影响优化不对齐的原因？

    - 有些模型训练时，会使用梯度累加策略，即累加到一定 step 数量之后才进行参数更新，这时在实现上需要注意对齐。
    - 在图像分类领域，大多数 Vision Transformer 模型都采用了 AdamW 优化器，并且会设置 weight decay，同时部分参数设置为 no weight decay，例如位置编码的参数通常设置为 no weight decay，no weight decay 参数设置不正确，最终会有明显的精度损失，需要特别注意。一般可以通过分析模型权重来发现该问题，分别计算 PyTorch 模型和飞桨模型每层参数权重的平均值、方差，对每一层依次对比，有显著差异的层可能存在问题，因为在 weight decay 的作用下，参数权重数值会相对较小，而未正确设置 no weight decay，则会造成该层参数权重数值异常偏小。设置 no weight decay 可以参照[这里](https://github.com/PaddlePaddle/PaddleClas/blob/release%2F2.3/ppcls/arch/backbone/model_zoo/resnest.py#L72)。
    - 在 OCR 识别等任务中，`Adadelta` 优化器常常被使用，该优化器与 PyTorch 实现目前稍有不同，但是不影响模型训练精度对齐。在做前反向对齐时，可以将该优化器替换为 Adam 等优化器（飞桨与参考代码均需要替换）；对齐完成之后，再使用 `Adadelta` 优化器进行完整的训练精度对齐。

3. 有没有什么任务不需要进行优化对齐的呢？

    在某些任务中，比如说深度学习可视化、可解释性等任务中，一般只要求模型前向过程，不需要训练，此时优化器、学习率等用于模型训练的模块对于该类模型迁移是不需要的。

4. 飞桨的学习率策略对不齐怎么办？

    - 飞桨中参数的学习率受到优化器学习率和 `ParamAttr` 中设置的学习率影响，在对齐时也需要保证这个部分的行为一致。
    - 有些网络的学习率策略比较细致，比如带 warmup 的学习率策略，这里需要保证起始学习率等参数都完全一致。

## 七、反向梯度对齐

反向梯度对齐的目的是确保迁移后的模型反向传播以及权重更新的行为与原始模型一致，同时也是对上一步*模型训练超参对齐*的验证。具体的检验方法是通过两次（或以上）迭代训练进行检查（这是因为第一次迭代之后，会根据反向传播计算得到的梯度与设定的训练超参更新模型参数，第二次迭代前向传播时，使用的是更新后的参数），若迁移前后的模型第二个迭代的训练 loss 一致，说明二者更新后的参数一致，则可以认为二者反向已对齐。

**【基本流程】**

此处可以通过 numpy 生成 fake data 和 label（推荐），也可以准备固定的真实数据。具体流程如下：

1. **检查训练超参**：检查两个代码的训练超参数全部一致，如优化器及其超参数、学习率、BatchNorm/LayerNorm 中的 eps 等。
2. **关闭随机操作**：将飞桨与 PyTorch 网络中涉及的所有随机操作全部关闭，如 dropout、drop_path 等，推荐将模型设置为 eval 模式（`model.eval()`）。
3. **训练并比较损失**：加载相同的 weight dict（可以通过 PyTorch 来存储随机的权重），将准备好的数据分别传入网络并迭代，观察二者 loss 是否一致（此处 batch-size 要一致，如果使用多个真实数据，要保证传入网络的顺序一致）。如果经过 2 次迭代以上，loss 均可以对齐，则基本可以认为反向对齐。

**【实战】**

本部分可以参考文档：[mobilenetv3_prod/Step1-5/README.md](https://github.com/PaddlePaddle/models/blob/release%2F2.2/tutorials/mobilenetv3_prod/Step1-5/README.md#4.5)。

**【核验】**

对于待迁移的项目，反向对齐核验流程如下。

1. 输入：fake data & label
2. 输出：

    - 飞桨/PyTorch：dict，key 为 tensor 的 name（自定义），value 为具体 loss 的值。最后将 dict 使用 reprod_log 保存到各自的文件中，建议命名为`losses_paddle.npy`和`losses_pytorch.npy`。
3. 自测：使用 reprod_log 加载 2 个文件，使用 report 功能，记录结果到日志文件中，建议命名为`losses_diff_log.txt`，观察 diff，二者 diff 小于特定的阈值即可。
4. 注意：

    - loss 需要保存至少 2 轮以上。
    - 在迭代的过程中，需要保证模型的 batch size 等超参数完全相同
    - 在迭代的过程中，需要设置`model.eval()`，使用固定的假数据，同时加载相同权重的预训练模型。

**【FAQ】**

1. 怎么打印反向梯度？

    - 飞桨打印反向和参数更新，可以参考[代码实例](https://github.com/jerrywgz/PaddleDetection/blob/63783c25ca12c8a7e1d4d5051d0888b64588e43c/ppdet/modeling/backbones/resnet.py#L598)。
    - PyTorch 打印反向和参数更新，可以参考[代码实例](https://github.com/jerrywgz/mmdetection/blob/ca9b8ef3e3770c4ad268a2fad6c55eb5d066e1b4/mmdet/models/backbones/resnet.py#L655)。

2. 反向没有对齐怎么办？

    反向对齐时，如果第一轮 loss 就没有对齐，则需要仔细先排查模型前向部分。

    如果第二轮开始，loss 开始无法对齐，则首先需要排查超参数的差异，超参数检查无误后，在`loss.backward()`方法之后，使用`tensor.grad`获取梯度值，二分的方法查找 diff，定位出飞桨与 PyTorch 梯度无法对齐的 API 或者操作，然后进一步验证。

    梯度的打印方法示例代码如下所示，注释内容即为打印网络中所有参数的梯度 shape。

    ```python
    # 代码地址：https://github.com/littletomatodonkey/AlexNet-Prod/blob/63184b258eda650e7a8b7f2610b55f4337246630/pipeline/Step4/AlexNet_paddle/train.py#L93
        loss_list = []
        for idx in range(max_iter):
            image = paddle.to_tensor(fake_data)
            target = paddle.to_tensor(fake_label)

            output = model(image)
            loss = criterion(output, target)
            loss.backward()
            # for name, tensor in model.named_parameters():
            #     grad = tensor.grad
            #     print(name, tensor.grad.shape)
            #     break
            optimizer.step()
            optimizer.clear_grad()
            loss_list.append(loss)
    ```

    如果只希望打印特定参数的梯度，可以用下面的方式。

    ```python
    import paddle

    def print_hook_fn(grad):
        print(grad)

    x = paddle.to_tensor([0., 1., 2., 3.], stop_gradient=False)
    h = x.register_hook(print_hook_fn)
    w = x * 4
    w.backward()
    # backward 之后会输出下面的内容
    #     Tensor(shape=[4], dtype=float32, place=CPUPlace, stop_gradient=False,
    #            [4., 4., 4., 4.])
    ```

## 八、训练集数据读取对齐

在 “三、小数据集数据读取对齐” 中，已经利用小数据集验证了数据读取对齐。但在读取用于训练的完整数据集时，可能仍会出现数据集版本等原因造成的差异。因此，建议使用完整数据集再次验证数据读取对齐，以确保训练时使用的数据是一致的。

**【基本流程】**

该部分内容与 “三、小数据集数据读取对齐” 内容基本一致，建议对照 [paddle.vision 高层 API 文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/vision/Overview_cn.html)，参考 PyTorch 的代码，实现训练集数据读取与预处理模块即可。

**【注意事项】**

该部分内容，可以参考 “七、反向梯度对齐” 的自测方法，将输入的`fake_data`与`fake_label`替换为训练的 dataloader，但是需要注意的是：

在使用 train dataloader 的时候，建议设置 random seed。

PyTorch 设置 seed 的方法：

```python
torch.manual_seed(config.SEED)
torch.cuda.manual_seed_all(config.SEED)
np.random.seed(config.SEED)
random.seed(config.SEED)
```

飞桨设置 seed 的方法：

```python
paddle.seed(config.SEED)
np.random.seed(config.SEED)
random.seed(config.SEED)
```

**【FAQ】**

1. 数据预处理时，报错信息过于复杂怎么办？

    在前向过程中，如果数据预处理过程运行出错，请先将 `paddle.io.DataLoader`的 `num_workers` 参数设为 0，然后根据单个进程下的报错日志定位出具体的 bug。

2. 数据读取无法对齐怎么办？

    - 数据读取需要注意图片读取方式是 opencv 还是 PIL.Image，图片格式是 RGB 还是 BGR，迁移时，需要保证迁移代码和参考代码完全一致。
    - 对于使用飞桨内置数据集，比如`paddle.vision.datasets.Cifar10`等，可能无法完全与参考代码在数据顺序上保持一致，如果是全量数据使用，对结果不会有影响，如果是按照比例选取子集进行训练，则建议重新根据参考代码实现数据读取部分，保证子集完全一致。
    - 如果数据处理过程中涉及到随机数生成，建议固定 seed (`np.random.seed(0)`, `random.seed(0)`)，查看迁移代码和参考代码处理后的数据是否有 diff。
    - 不同的图像预处理库，使用相同的插值方式可能会有 diff，建议使用相同的库对图像进行 resize。
    - 视频解码时，不同库解码出来的图像数据会有 diff，注意区分解码库是 opencv、decord 还是 pyAV，需要保证迁移代码和 PyTorch 代码完全一致。

## 九、网络初始化对齐

本部分对齐建议对照飞桨初始化 API 文档与参考代码的初始化实现对齐。

考虑到向下兼容，飞桨的权重初始化定义方式与 PyTorch 存在区别。在 PyTorch 中，通常调用 `torch.nn.init`下的函数来定义权重初始化方式，例如：

```python
for m in self.modules():
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0, 0.01)
        nn.init.zeros_(m.bias)
```

而飞桨提供的初始化方式为在定义模块时直接修改 API 的`ParamAttr`属性，与`torch.nn.init`等系列 API 的使用方式不同，例如：

```python
self.conv = nn.Conv2D(in_channels, out_channels, kernel_size, weight_attr=nn.initializer.KaimingNormal())
self.fc = paddle.nn.Linear(4096, 512,
            weight_attr=init.Normal(0, 0.01),
            bias_attr=init.Constant(0))
```

下面给出了部分初始化 API 的映射表。

| **飞桨  API**                        | **PyTorch API**                |
| ------------------------------------ | ------------------------------ |
| paddle.nn.initializer.KaimingNormal  | torch.nn.init.kaiming_normal_  |
| paddle.nn.initializer.KaimingUniform | torch.nn.init.kaiming_uniform_ |
| paddle.nn.initializer.XavierNormal   | torch.nn.init.xavier_normal_   |
| paddle.nn.initializer.XavierUniform  | torch.nn.init.xavier_uniform_  |

更多初始化 API 可以参考[PyTorch 初始化 API 文档](https://pytorch.org/docs/stable/nn.init.html)以及[飞桨初始化 API 文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/Overview_cn.html#chushihuaxiangguan)。

**【FAQ】**

1、使用相同的分布初始化模型还是不能对齐怎么办？

    对于不同的深度学习框架，网络初始化在大多情况下，即使值的分布完全一致，也无法保证值完全一致，这也是迁移中不确定性较大的地方。如果十分怀疑初始化导致的问题，建议将参考的初始化权重转成飞桨模型，加载该初始化模型训练，检查收敛精度。

2、如何对齐 torch.nn.init.constant_() ？

    飞桨中目前没有 `torch.nn.init.constant_()` 的方法，如果希望对参数赋值为常数，可以使用 `paddle.nn.initializer.Constant` API；或者可以参考下面的实现。更加具体的解释可以参考：[模型参数初始化对齐](https://github.com/PaddlePaddle/models/blob/release/2.3/tutorials/article-implementation/initializer.md)。

    ```python
    import paddle
    import paddle.nn as nn
    import numpy as np
    # Define the linear layer.
    m = paddle.nn.Linear(2, 4)
    print(m.bias)
    if isinstance(m, nn.Layer):
        print("set m.bias")
        m.bias.set_value(np.ones(shape=m.bias.shape, dtype="float32"))
        print(m.bias)
    ```

3、初始化是怎么影响不同类型的模型的？

    - 对于 CNN 模型而言，模型初始化对最终的收敛精度影响较小，在迭代轮数与数据集足够的情况下，最终精度指标基本接近；
    - Transformer、图像生成等模型对于初始化比较敏感，在这些模型训练对齐过程中，建议对初始化进行重点检查；
    - 领域自适应算法由于需要基于初始模型生成伪标签，因此对初始网络敏感，建议加载预训练的模型进行训练。

## 十、模型训练对齐

### 10.1 规范训练日志

训练过程中，损失函数(`loss`)可以直接反映目前网络的收敛情况，数据耗时(`reader_cost`)对于分析 GPU 利用率非常重要，一个 batch 训练耗时(`batch_cost`)对于判断模型的整体训练时间非常重要，因此希望在训练中添加这些统计信息，便于分析模型的收敛和资源利用情况。

**【基本流程】**

在训练代码中添加日志统计信息，对训练中的信息进行统计。

    - 必选项：损失值`loss`, 训练耗时`batch_cost`, 数据读取耗时`reader_cost`。
    - 建议项：当前`epoch`, 当前迭代次数`iter`，学习率(`lr`), 准确率(`acc`)等。

如果训练中同时包含评估过程，则需要在日志里添加模型的`评估结果`信息。

```plain
[2021/12/04 05:16:13] root INFO: [epoch 0, iter 0][TRAIN]avg_samples: 32.0 , avg_reader_cost: 0.0010543 sec, avg_batch_cost: 0.0111100 sec, loss: 0.3450000 , avg_ips: 2880.2952878 images/sec
[2021/12/04 05:16:13] root INFO: [epoch 0, iter 0][TRAIN]avg_samples: 32.0 , avg_reader_cost: 0.0010542 sec, avg_batch_cost: 0.0111101 sec, loss: 0.2450000 , avg_ips: 2880.2582019 images/sec
```

**【注意事项】**

- 日志打印与保存也会占用少量时间，这里不建议统计其耗时，防止对统计结果造成影响。
- `autolog`支持训练和预测的日志规范化，更多关于`autolog`的使用可以参考：https://github.com/LDOUBLEV/AutoLog。

**【实战】**

参考代码：[mobilenetv3_prod/Step1-5/mobilenetv3_ref/train.py](https://github.com/PaddlePaddle/models/blob/release/2.2/tutorials/mobilenetv3_prod/Step1-5/mobilenetv3_ref/train.py)。

具体地，规范化的训练日志应包含读取数据用时、训练用时、样本数以及训练准确率等信息，可以参考如下所示的方式实现。

```python
def train_one_epoch(model,
                    criterion,
                    optimizer,
                    data_loader,
                    epoch,
                    print_freq):
    model.train()
    # 读取数据总用时
    train_reader_cost = 0.0
    # 训练总用时
    train_run_cost = 0.0
    # 总样本数
    total_samples = 0
    # 准确率
    acc = 0.0
    # 读取数据开始时间
    reader_start = time.time()
    batch_past = 0

    for batch_idx, (image, target) in enumerate(data_loader):
        # 读取数据用时
        train_reader_cost += time.time() - reader_start
        # 记录训练开始的时间
        train_start = time.time()
        # 前向传播
        output = model(image)
        loss = criterion(output, target)
        # 反向传播
        loss.backward()
        # 更新参数
        optimizer.step()
        optimizer.clear_grad()
        # 训练用时
        train_run_cost += time.time() - train_start
        # 记录训练准确率
        acc = utils.accuracy(output, target).item()
        total_samples += image.shape[0]
        batch_past += 1

        if batch_idx > 0 and batch_idx % print_freq == 0:
            # 每隔 print_freq 输出一次
            msg = "[Epoch {}, iter: {}] acc: {:.5f}, lr: {:.5f}, loss: {:.5f}, avg_reader_cost: {:.5f} sec, avg_batch_cost: {:.5f} sec, avg_samples: {}, avg_ips: {:.5f} images/sec.".format(
                epoch, batch_idx, acc / batch_past,
                optimizer.get_lr(),
                loss.item(), train_reader_cost / batch_past,
                (train_reader_cost + train_run_cost) / batch_past,
                total_samples / batch_past,
                total_samples / (train_reader_cost + train_run_cost))
            # 仅需在第一个 device 上 log
            if paddle.distributed.get_rank() <= 0:
                print(msg)
            sys.stdout.flush()
            train_reader_cost = 0.0
            train_run_cost = 0.0
            total_samples = 0
            acc = 0.0
            batch_past = 0

        reader_start = time.time()
```

### 10.2 训练精度对齐

**【基本流程】**

完成前面的步骤之后，就可以开始全量数据的训练对齐任务了。按照下面的步骤可以进行训练对齐：

1. 准备 train/eval data，loader，model。
2. 对 model 按照 PyTorch 项目所述进行初始化(如果 PyTorch 项目训练中加载了预训练模型，则这里也需要保持一致)。
3. 加载配置，开始训练，迭代得到最终模型与评估指标，将评估指标使用 reprod_log 保存到文件中。
4. 将飞桨提供的参考指标使用 reprod_log 提交到另一个文件中。
5. 对于单一指标（如 Top1 acc），可以直接检查指标是否符合预期，如果同时输出多个指标，建议使用 reprod_log 排查 diff，小于阈值，即可完成自测。

**【实战】**

本部分可以参考代码：[mobilenetv3_prod/Step6/train.py#L371](https://github.com/PaddlePaddle/models/blob/release%2F2.2/tutorials/mobilenetv3_prod/Step6/train.py#L371)。

**【核验】**

对于待迁移的项目，训练对齐核验流程如下：

1. 输入：train/eval dataloader, model
2. 输出：

    - 飞桨：dict，key 为保存值的 name（自定义），value 为具体评估指标的值。最后将 dict 使用 reprod_log 保存到文件中，建议命名为`train_align_paddle.npy`。
    - benchmark：dict，key 为保存值的 name（自定义），value 为模型迁移评估指标要求的值。最后将 dict 使用 reprod_log 保存到文件中，建议命名为`train_align_benchmark.npy`。

3. 自测：对于单一指标（如 Top1 acc），可以直接检查指标是否符合预期；如果是多个指标，可以使用 reprod_log 加载 2 个训练结果文件，使用 report 功能，记录结果到日志文件中，建议命名为`train_align_diff_log.txt`，观察 diff，二者 diff 小于特定的阈值即可。以基于 ImageNet1k 分类数据集或者 COCO2017 检测数据集的模型训练为例，精度误差在 0.1%或者以内是可以接受的。

**【FAQ】**

1. 训练过程怎么更好地对齐？

    - 若条件允许，迁移工作之前建议先基于 PyTorch 代码完成训练，保证与 PyTorch 的代码训练精度符合预期，并且将训练策略和训练过程中的关键指标记录保存下来，比如每个 epoch 的学习率、Train Loss、Eval Loss、Eval Acc 等，在迁移网络的训练过程中，将关键指标保存下来，这样可以将飞桨与 PyTorch 训练中关键指标的变化曲线绘制出来，能够很方便地进行对比；
    - 若使用较大的数据集训练，一次完整训练的成本较高，此时可以隔一段时间查看精度。如果精度差异较大，建议先停止实验，排查原因。

2. 如果训练过程中出现不收敛的情况，怎么办？

    - 简化网络和数据，实验是否收敛；
    - 如果是基于原有实现进行改动，可以尝试控制变量法，每次做一个改动，逐个排查；
    - 检查学习率是否过大、优化器设置是否合理，排查 weight decay 是否设置正确；
    - 保存不同 step 之间的模型参数，观察模型参数是否更新。

3. 如果训练的过程中出 nan 怎么办？

    一般是因为除 0 或者 log0 的情况，可以重点检查以下部分：
    1. 如果使用了预训练模型，检查预训练模型是否加载正确
    2. 确认 reader 的预处理中是否会出现空的 tensor（检测任务中一般会出现该问题）
    3. 模型结构中计算 loss 的部分是否有考虑到正样本为 0 的情况
    4. 该问题也可能是某个 API 的数值越界导致的，可以测试较小的输入是否仍会出现 nan。

4. 其他细分场景下有什么导致训练不对齐的原因？

    - 小数据上指标波动可能比较大。若时间允许，可以进行多次实验，取平均值。
    - Transformer 系列模型，在模型量级比较小的情况下，使用更大的 batch size 以及对应的学习率进行训练往往会获得更高的精度。在迁移时，建议保证 batch size 和学习率完全一致，否则即使精度对齐了，也可能会隐藏其他没有对齐的风险项。
    - 目标检测、图像分割等任务中，训练通常需要加载 backbone 的权重作为预训练模型。注意在训练对齐时，需要加载经过转换的 backbone 权重。
    - 生成任务中，训练时经常需要固定一部分网络参数。对于一个参数`param`，可以通过 `param.trainable = False` 来固定。
    - 在训练 GAN 时，通常通过 GAN 的 loss 较难判断出训练是否收敛。建议每训练几次迭代保存一次训练生成的图像，通过可视化判断训练是否收敛。
    - 在训练 GAN 时，如果飞桨实现的代码已经可以与参考代码完全一致，参考代码和飞桨代码均难以收敛，则可以在训练时检查 loss。如果 loss 大于一个阈值或者为 NAN，说明训练不收敛，应终止训练，使用最新保存的参数重新继续训练。可以参考该链接的实现：[链接](https://github.com/JennyVanessa/Paddle-GI)。

5. 怎样设置运行设备?

飞桨通过 `paddle.set_device` 函数（全局，会决定 Dataloader、模型组网、to_tensor 等操作产出的数据所在的设备）来确定模型运行的设备。而 PyTorch 则是通过 `model.to("device")` （局部，这句话仅影响 model 所在的设备）来确定模型的运行设备。

## 十一、预测程序验证

**【基本流程】**

模型训练完成之后，使用该模型基于训练引擎进行预测，主要包含：

1. 定义模型结构，加载模型权重。
2. 加载图像，对其进行数据预处理。
3. 模型预测。
4. 对模型输出进行后处理，获取最终输出结果。

**【注意事项】**

在模型评估过程中，为了保证数据可以组成图片尺寸相同的 batch，一般会使用 resize/crop/padding 等方法以保持尺度的一致性。

在预测推理过程中，需要注意 crop 是否合适，比如 OCR 识别任务中，crop 的操作会导致识别结果不全。

**【实战】**

MobileNetV3 的预测程序：[Step6/tools/predict.py](https://github.com/PaddlePaddle/models/blob/release%2F2.2/tutorials/mobilenetv3_prod/Step6/tools/predict.py)。主要源代码如下所示：

```python
@paddle.no_grad()
def main(args):
    # 模型定义
    model = paddlevision.models.__dict__[args.model](
        pretrained=args.pretrained, num_classes=args.num_classes)

    model = nn.Sequential(model, nn.Softmax())
    model.eval()

    # 定义图片变换
    eval_transforms = ClassificationPresetEval(args.resize_size,
                                               args.crop_size)
    # 要预测的图片
    with open(args.img_path, 'rb') as f:
        img = Image.open(f).convert('RGB')

    # 变换图片
    img = eval_transforms(img)
    # 转换为 Tensor
    img = paddle.to_tensor(img)
    # 扩展 batch 维度, batch_size=1
    img = img.expand([1] + img.shape)

    # 预测的各类别的概率值
    output = model(img).numpy()[0]

    # 概率值最大的类别
    class_id = output.argmax()
    # 对应的概率值
    prob = output[class_id]
    print(f"class_id: {class_id}, prob: {prob}")
    return output
```

**【核验】**

预测程序按照文档中的命令操作可以正常跑通，文档中给出预测结果可视化结果或者终端输出结果。

## 十二、单机多卡训练

单机多卡训练使用一台机器上的多个 GPU 并行进行训练，可以提升训练效率。与单机单卡相比，在单机多卡的条件下训练模型，需要对以下几个方面进行修改：

- 多卡并行数据读取
- 多卡并行训练环境及模型初始化
- 模型保存、日志保存
- 训练启动指令

**【基本流程】**

若要将单机单卡训练的代码修改为多机多卡训练的代码，需要：

- 修改数据读取代码，将`paddle.io.RandomSampler`改为`paddle.io.DistributedRandomSampler`
- 初始化多卡并行训练环境，并使用`paddle.DataParallel()`对模型进行封装
- 模型保存、日志保存的代码需要增加对卡号的判断，仅在 0 号卡上保存即可

**【实战】**

单机多卡的代码实现请参考：[链接](https://github.com/PaddlePaddle/models/blob/release%2F2.3/tutorials/mobilenetv3_prod/Step6/train.py)。单机单卡属于单机多卡的特殊情形（卡数为 1），因此也适用于单机单卡的训练过程。

### 12.1 数据读取

多卡数据读取变化在于 sampler。对于单机单卡，使用 `paddle.io.RandomSampler` 作为 sampler 用于组成 batch。

```python
train_sampler = paddle.io.RandomSampler(dataset)
train_batch_sampler = paddle.io.BatchSampler(
    sampler=train_sampler, batch_size=args.batch_size)
```

对于单机多卡任务，需要使用 `paddle.io.DistributedBatchSampler`。

```python
train_batch_sampler = paddle.io.DistributedBatchSampler(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False
    )
```

**【注意事项】**

在这种情况下，单机多卡的代码仍然能够以单机单卡的方式运行，因此建议以单机多卡的 sampler 方式进行迁移。

### 12.2 多卡模型初始化

如果以多卡的方式运行，需要初始化并行训练环境。代码如下所示：

```python
# 如果当前节点用于训练的设备数量大于 1，则初始化并行训练环境
if paddle.distributed.get_world_size() > 1:
    paddle.distributed.init_parallel_env()
```

在模型组网并初始化参数之后，需要使用`paddle.DataParallel()`对模型进行封装，使得模型可以通过数据并行的模式被执行。代码如下所示：

```python
if paddle.distributed.get_world_size() > 1:
    model = paddle.DataParallel(model)
```

### 12.3 模型保存、日志保存等其他模块

以模型保存为例，您只需要在 0 号卡上保存即可，否则多个 trainer 同时保存可能会造成写冲突，导致最终保存的模型不可用。

```python
if paddle.distributed.get_rank() == 0:
    paddle.save(params_state_dict, model_path + ".pdparams")
```

### 12.4 程序启动方式

对于单机单卡，启动脚本如下所示：

```shell
export CUDA_VISIBLE_DEVICES=0
python3 train.py \
    --data-path /paddle/data/ILSVRC2012_torch \
    --lr 0.00125 \
    --batch-size 32 \
    --output-dir "./output/"
```

对于单机多卡（示例中为 8 卡训练），启动脚本如下所示：

```shell
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python3 -m paddle.distributed.launch \
    --gpus="0,1,2,3,4,5,6,7" \
    train.py \
    --data-path /paddle/data/ILSVRC2012_torch \
    --lr 0.01 \
    --batch-size 32 \
    --output-dir "./output/"
```



**【注意事项】**

这里 8 卡训练时，虽然单卡的 batch size 没有变化 (32)，但是总的 batch size 相当于是单卡的 8 倍，因此学习率也设置为了单卡时的 8 倍。

**【实战】**

本部分可以参考文档：[单机多卡训练脚本](https://github.com/littletomatodonkey/AlexNet-Prod/blob/master/pipeline/Step5/AlexNet_paddle/shell/train_dist.sh)。

train.py 主要源代码：

```python
def main(args):
    if args.output_dir:
        utils.mkdir(args.output_dir)

    print(args)

    device = paddle.set_device(args.device)

    # 多卡训练环境初始化
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()

    # 数据读取定义
    train_dir = os.path.join(args.data_path, 'train')
    val_dir = os.path.join(args.data_path, 'val')
    dataset, dataset_test, train_sampler, test_sampler = load_data(
        train_dir, val_dir, args)
    train_batch_sampler = train_sampler
    data_loader = paddle.io.DataLoader(
        dataset=dataset,
        num_workers=args.workers,
        return_list=True,
        batch_sampler=train_batch_sampler)
    test_batch_sampler = paddle.io.BatchSampler(
        sampler=test_sampler, batch_size=args.batch_size)
    data_loader_test = paddle.io.DataLoader(
        dataset_test,
        batch_sampler=test_batch_sampler,
        num_workers=args.workers)

    print("Creating model")
    model = paddlevision.models.__dict__[args.model](
        pretrained=args.pretrained)

    # 损失函数定义
    criterion = nn.CrossEntropyLoss()

    lr_scheduler = paddle.optimizer.lr.StepDecay(
        args.lr, step_size=args.lr_step_size, gamma=args.lr_gamma)

    # 优化器定义
    opt_name = args.opt.lower()
    if opt_name == 'sgd':
        optimizer = paddle.optimizer.Momentum(
            learning_rate=lr_scheduler,
            momentum=args.momentum,
            parameters=model.parameters(),
            weight_decay=args.weight_decay)
    elif opt_name == 'rmsprop':
        optimizer = paddle.optimizer.RMSprop(
            learning_rate=lr_scheduler,
            momentum=args.momentum,
            parameters=model.parameters(),
            weight_decay=args.weight_decay,
            eps=0.0316,
            alpha=0.9)
    else:
        raise RuntimeError(
            "Invalid optimizer {}. Only SGD and RMSprop are supported.".format(
                args.opt))

    # 从 checkpoints 恢复训练
    if args.resume:
        layer_state_dict = paddle.load(os.path.join(args.resume, '.pdparams'))
        model.set_state_dict(layer_state_dict)
        opt_state_dict = paddle.load(os.path.join(args.resume, '.pdopt'))
        optimizer.load_state_dict(opt_state_dict)

    # 多卡训练模型初始化
    if paddle.distributed.get_world_size() > 1:
        model = paddle.DataParallel(model)

    if args.test_only and paddle.distributed.get_rank() == 0:
        top1 = evaluate(model, criterion, data_loader_test, device=device)
        return top1

    print("Start training")
    start_time = time.time()
    best_top1 = 0.0

    # 从 start_epoch 开始训练 epochs 轮
    for epoch in range(args.start_epoch, args.epochs):
        train_one_epoch(model, criterion, optimizer, data_loader, device,
                        epoch, args.print_freq)
        lr_scheduler.step()
        # 保存模型及优化器参数，仅需在 0 号卡上保存
        if paddle.distributed.get_rank() == 0:
            top1 = evaluate(model, criterion, data_loader_test, device=device)
            best_top1 = max(best_top1, top1)
            if args.output_dir:
                paddle.save(model.state_dict(),
                            os.path.join(args.output_dir,
                                         'model_{}.pdparams'.format(epoch)))
                paddle.save(optimizer.state_dict(),
                            os.path.join(args.output_dir,
                                         'model_{}.pdopt'.format(epoch)))
                paddle.save(model.state_dict(),
                            os.path.join(args.output_dir, 'latest.pdparams'))
                paddle.save(optimizer.state_dict(),
                            os.path.join(args.output_dir, 'latest.pdopt'))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    return best_top1
```

## 十三、常见 bug 汇总

在论文复现中，可能因为各种原因出现报错，下面列举了常见的问题和解决方法，从而提供 debug 的方向：

### 13.1 显存泄漏

若发生显存泄漏，在 `nvidia-smi` 等命令下，可以明显地观察到显存的增加，最后会因为 `out of memory` 的错误而程序终止。

**【可能原因】**

Tensor 切片的时候增加变量引用，导致显存增加。

**【解决方法】**

使用 where, gather 函数替代原有的 slice 方式。

```python
a = paddle.range(3)
c = paddle.ones([3])
b = a>1
# 会增加引用的一种写法
c[b] = 0
# 修改后
paddle.where(b, paddle.zeros(c.shape), c)
```

### 13.2 内存泄漏

内存泄漏和显存泄漏相似，并不能立即察觉，而是在使用 `top` 命令时，观察到内存显著增加，最后会因为 `can't allocate memory` 的错误而程序终止，如图所示是 `top` 命令下观察内存变化需要检查的字段。

![img](../../images/memory_leak.png)


**【可能原因】**

1. 对在计算图中的 tensor 进行了不需要的累加、保存等操作，导致反向传播中计算图没有析构，解决方法如下：
2. **预测阶段**：在 predict 函数上增加装饰器@paddle.no_grad()；在预测部分的代码前加上 with paddle.no_grad()。
3. **训练阶段**：对于不需要进行加入计算图的计算，将 tensor detach 出来；对于不需要使用 tensor 的情形，将 tensor 转换为 numpy 进行操作。例如下面的代码：

```python
cross_entropy_loss = paddle.nn.CrossEntropyLoss()
loss = cross_entropy_loss(pred, gt)
# 会导致内存泄漏的操作
loss_total += loss
# 修改后
loss_total += loss.numpy() # 如果可以转化为 numpy
loss_total += loss.detach().clone() # 如果需要持续使用 tensor
```

**【排查方法】**

1. 在没有使用 paddle.no_grad 的代码中，寻找对模型参数和中间计算结果的操作。
2. 考虑这些操作是否应当加入计算图中（即对最后损失产生影响）。
3. 如果不需要，则需要对操作中的参数或中间计算结果进行`.detach().clone()`或者`.numpy()` 后操作。

### 13.3 dataloader 加载数据时间长

**【解决方法】**

增大 num_worker 的值，提升数据 IO 速度，一般建议设置 4 或者 8。

### 13.4 报错信息不明确

**【解决方法】**

（1）可以设置环境变量 export GLOG_v=4（默认为 0），打印更多的日志信息（该值越大，信息越多，一般建议 4 或 6 即可）。

（2）如果是单机多卡训练报错，可以前往 log 目录下寻找 worklog.x 进行查看，其中 worklog.x 代表第 x 卡的报错信息。
