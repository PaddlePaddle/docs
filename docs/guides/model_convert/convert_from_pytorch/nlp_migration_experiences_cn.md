# NLP - 迁移经验汇总

这里提供 NLP 各个方向从 PyTorch 迁移到飞桨的基本流程、常用工具、定位问题的思路及解决方法。

## 一、迁移概述

模型迁移本质上需要从模型训练与预测的角度去完成该任务，保证训练结果与预测结果与参考代码保持一致。

- 在模型预测过程中，模型组网、数据处理与加载、评估指标等均需要严格对齐，否则无法产出与参考代码完全相同的模型及预测结果。
- 在训练阶段，除了模型结构等元素，训练的损失函数、梯度、训练超参数、初始化方法以及训练精度也是我们需要迁移并对齐的内容。一个完整的模型训练包括定义模型结构并初始化，将处理后的数据送进网络，对输出的内容与真值计算损失函数，并反向传播与迭代的过程。

### 1.1 迁移流程

本章节从模型训练和推理需要的基本操作出发，对迁移工作进行任务分解，如下图所示。同时对每个环节进行对齐验证，检查每个环节飞桨和 Pytorch 模型在同样输入下的输出是否一致，以便快速发现问题，降低问题定位的难度。



<p align="center">
  <img src="https://raw.githubusercontent.com/ymyjl/docs/torch_migrate/docs/guides/model_convert/pictures/porcess.png" align="middle"  width="500" />
</p>


1. **迁移准备**：迁移工作首先需要安装必要的软件和工具（包括飞桨、PyTorch 或 TensorFlow 的安装、差异核验工具的安装等），然后准备要迁移的模型以及使用的数据集，同时了解源代码结构、跑通模型训练，最后对源代码进行解析，统计缺失算子。
2. **模型前向对齐**：这是迁移工作最基本的部分。在搭建神经网络时，会使用到框架提供的 API（内置的模块、函数等）。在迁移时，需要对这些 API 转换成飞桨对应的 API。转换前后的模型应具有相同的网络结构，使用相同的模型权重时，对于相同的输入，二者的输出结果应该一致。有时同一个神经网络有不同的版本、同一个版本有不同的实现方式或者在相同的神经网络下使用不同的超参，这些差别会对最终的收敛精度和性能造成一定影响。通常，我们以神经网络作者本身的实现为准，也可以参考不同框架（例如飞桨、TensorFlow、PyTorch 等）的官方实现或其他主流开源工具箱（例如 MMDetection）。PyTorch 的大部分 API 在飞桨中可找到对应 API，可以直接对模型组网部分代码涉及的 API 进行手动转换。为了判断转换后的飞桨模型组网能获得和 Pytorch 参考实现同样的输出，可将两个模型参数固定，并输入相同伪数据，观察两者的产出差异是否在阈值内。
3. **小数据集数据读取对齐**：数据读取对齐为了验证数据加载、数据预处理、数据增强与原始代码一致。为了快速验证数据读取对齐，建议准备一个小数据集（训练集和验证集各 8~16 张图像即可，压缩后数据大小建议在`20M`以内）。
4. **评估指标对齐**：评估指标是模型精度的度量。在计算机视觉中，不同任务使用的评估指标有所不同（比如，在图像分类任务中，常用的指标是 Top-1 准确率与 Top-5 准确率；在图像语义分割任务中，常用的指标是 mIOU）为了检验迁移后的模型能否达到原模型的精度指标，需要保证使用的评估指标与原始代码一致以便对照。飞桨提供了一系列 Metric 计算类，而 PyTorch 中目前可以通过组合的方式实现或者调用第三方的 API。
5. **损失函数对齐**：损失函数是训练模型时的优化目标，使用的损失函数会影响模型的精度。在模型迁移时，需要保证迁移后模型训练时使用的损失函数与原始代码中使用的损失函数一致，以便二者对照。飞桨与 PyTorch 均提供了常用的损失函数。
6. **模型训练超参对齐**：模型的训练超参包括学习率、优化器、正则化策略等。这些超参数指定了模型训练过程中网络参数的更新方式，训练超参数的设置会影响到模型的收敛速度及收敛精度。同样地，在模型迁移时，需要保证迁移前后模型使用的训练超参数一致，以便对照二者的收敛情况。飞桨中的 optimizer 有`paddle.optimizer`等一系列实现，PyTorch 中则有`torch.optim`等一系列实现。完成超参对齐后，可以使用反向梯度对齐统一验证该模块的正确性。
7. **反向梯度对齐**：在完成前向对齐的基础上，还需进行反向梯度对齐。反向梯度对齐的目的是确保迁移后的模型反向传播以及权重更新的行为与原始模型一致，同时也是对上一步*模型训练超参对齐*的验证。具体的检验方法是通过两次（或以上）迭代训练进行检查，若迁移前后的模型第二轮训练的 loss 一致，则可以认为二者反向已对齐。
8. **训练集数据读取对齐**：相同的神经网络使用不同的数据训练和测试得到的结果往往会存在差异。因此，为了能复现原始代码的精度，需要保证使用的数据完全相同，包括数据集的版本、使用的数据预处理方法和流程、使用的数据增强方式等。
9. **网络初始化对齐**：对于不同的深度学习框架，网络初始化在大多情况下，即使值的分布完全一致，也无法保证值完全一致，这里也是模型迁移不确定性比较大的地方。CNN 对于模型初始化相对来说没有那么敏感，在迭代轮数与数据集足够的情况下，最终精度指标基本接近。而 transformer 系列模型、超分模型、领域自适应算法对于初始化比较敏感，需要对初始化进行重点检查。如果十分怀疑初始化导致的问题，建议将参考的初始化权重转成飞桨模型权重，加载该初始化模型训练，检查收敛精度。
10. **训练精度对齐**：模型训练的最终结果是为了得到一个精度达标的模型。不同的框架版本、是否为分布式训练等可能会对训练精度有影响，在迁移前需要分析清楚对标的框架、硬件等信息。对比迁移前后模型的训练精度，若二者的差值在可以接受的误差范围内，则精度对齐完成。同时，如果在相同的硬件条件下，迁移前后的模型训练速度应接近。若二者差异非常大，则需要排查原因。
11. **模型预测验证**：模型训练完成之后，需要使用测试集对该模型基于训练引擎进行预测，确认预测结果与实际一致。

其中，2~5 是迁移的重点，其他模块比如：反向梯度、优化器、学习率生成等，要么本身结构单一，要么依赖已开发完成的网络结果才能和对标脚本形成对比。而且这些模块的脚本开发难度更小一些。

**【注意事项】**

如果遇到迁移时间较长的项目，建议：

- 根据自己的时间、资源、战略部署评估是否进行此项目迁移。
- 在决定迁移的情况下，参照本迁移指南中的对齐操作对模型、数据、优化方式等对齐，以最快的时间排除问题。
- 模型的实现具有相通性，为提升模型迁移效率，可参考和借鉴已实现模型的代码。飞桨提供了大规模的官方模型库，包含经过产业实践长期打磨的主流模型以及在国际竞赛中的夺冠模型，算法总数超过 500 多个，详细请参考链接：https://www.paddlepaddle.org.cn/modelbase。

**【获取更多飞桨信息】**

可以通过[API](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/index_cn.html)文档了解飞桨各接口的相关信息；还可通过[教程](https://aistudio.baidu.com/aistudio/course/introduce/1297)系统掌握如何使用飞桨进行训练、调试、调优、推理。

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

```plain
PaddlePaddle is installed successfully! Let's start deep learning with PaddlePaddle now.
```



**【FAQ】**

如何安装飞桨的 develop 版本？

在飞桨修复了框架的问题或者新增了 API 和功能之后，若需要立即使用，可以采用以下方式安装最新的 develop 版本：

进入[飞桨官网](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/develop/install/pip/linux-pip.html)，选择 develop 版本，并根据自己的情况选择其他字段，根据生成的安装信息安装，当选择 Linux-pip-CUDA10.2 字段后，就可以按照下面的信息安装。

```shell
python -m pip install paddlepaddle-gpu==0.0.0.post102 -f https://www.paddlepaddle.org.cn/whl/linux/gpu/develop.html
```

####

在对齐验证的流程中，我们依靠 reprod_log 差异核验工具查看飞桨和 PyTorch 同样输入下的输出是否相同，这样的查看方式具有标准统一，比较过程方便等优势。

Reprod_log 是一个用于 numpy 数据记录和对比工具，通过传入需要对比的两个 numpy 数组就可以在指定的规则下得到数据之差是否满足期望的结论。其主要接口的说明可以查看其 [GitHub 主页](https://github.com/PaddlePaddle/models/tree/release/2.3/tutorials/reprod_log)。

安装 reprod_log 的命令如下：

```bash
pip3 install reprod_log --force-reinstall
```

##### 1.2.1.2 安装 PyTorch

对于 PyTorch 的安装，请参阅 [PyTorch 官网](https://pytorch.org/get-started/locally/)，选择操作系统和 CUDA 版本，使用相应的命令安装。

运行 Python，输入下面的命令，如果可以正常输出，则说明 PyTorch 安装成功。

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

下面基于前面的示例代码，给出如何使用该工具。

文件夹中包含 write_log.py 和 check_log_diff.py 文件，其中 write_log.py 中给出了 ReprodLogger 类的使用方法，check_log_diff.py 给出了 ReprodDiffHelper 类的使用方法，依次运行两个 python 文件，使用下面的方式运行代码。

```plain
进入文件夹
cd pipeline/reprod_log_demo
随机生成矩阵，写入文件中
python3.7 write_log.py
进行文件对比，输出日志
python3.7 check_log_diff.py
```



最终会输出以下内容：

```plain
2021-09-28 01:07:44,832 - reprod_log.utils - INFO - demo_test_1:
2021-09-28 01:07:44,832 - reprod_log.utils - INFO -     mean diff: check passed: True, value: 0.0
2021-09-28 01:07:44,832 - reprod_log.utils - INFO - demo_test_2:
2021-09-28 01:07:44,832 - reprod_log.utils - INFO -     mean diff: check passed: False, value: 0.3336232304573059
2021-09-28 01:07:44,832 - reprod_log.utils - INFO - diff check failed
```



可以看出：对于 key 为 demo_test_1 的矩阵，由于 diff 为 0，小于设置的阈值 1e-6，核验成功；对于 key 为 demo_test_2 的矩阵，由于 diff 为 0.33，大于设置的阈值 1e-6，核验失败。

**【reprod_log 在迁移中的应用】**

针对迁移场景，基于 reprod_log 的结果记录模块，产出下面若干文件：

```plain
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

基于 reprod_log 的 ReprodDiffHelper 模块，产出下面 5 个日志文件。

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
  - 准备伪输入数据（fake input data）以及伪标签（fake label）：通过运行生成伪数据的参考代码：https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/torch_migration/pipeline/fake_data/gen_fake_data.py。

```plain
def gen_fake_data():
    fake_data = np.random.randint(1, 30522, size=(4, 64)).astype(np.int64)
    fake_label = np.array([0, 1, 1, 0]).astype(np.int64)
    np.save("fake_data.npy", fake_data)
    np.save("fake_label.npy", fake_label)
```





#### 1.2.3 分析并运行参考代码

需在特定设备(CPU/GPU)上，利用少量伪数据，跑通参考代码的预测过程(前向)以及至少 2 轮(iteration)迭代过程，用于生成和迁移代码进行对比的结果。

对于复杂的神经网络，完整的训练需要耗时几天甚至几个月，如果仅以最终的训练精度和结果做参考，会极大地降低开发效率。因此，我们可以利用少量数据进行少量迭代训练缩短时间（该迭代是执行了数据预处理、权重初始化、正向计算、loss 计算、反向梯度计算和优化器更新之后的结果，覆盖了网络训练的全部环节），并以此为对照展开后续的开发工作。

PyTorch 的实现：

https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/torch_migration/pipeline/Step5/bert_torch



项目的目录结构如下：

```plain
bert_torch
    |-accuracy.py                                 # 评价指标
    |-train.py                                    # 模型训练代码
    |-utils.py                                    # 工具类及函数
    |-log.log                                     #日志记录
    |-glue.py                                     #数据生成代码
    |-train.sh                                    #启动训练的 bash 脚本
```

为了便于实例的演示，可将以下参考代码下载到本地。

```plain
# 克隆参考代码所在的项目 repo 到本地
git clone https://github.com/PaddlePaddle/PaddleNLP.git
cd examples/torch_migration/
```



#### 1.2.4 构建迁移项目

为了便于对比，建议按照 PyTorch 项目的结构构建飞桨项目。例如，对于上述的 BERT 项目的复现，按照原始项目的结构，可构建飞桨项目如下：

```python
bert_torch
    |-accuracy.py                                 # 评价指标
    |-train.py                                    # 模型训练代码
    |-utils.py                                    # 工具类及函数
    |-log.log                                     #日志记录
    |-glue.py                                     #数据生成代码
    |-train.sh                                    #启动训练的 bash 脚本
```

#### 1.2.5 评估算子和 API

使用飞桨搭建神经网络流程与 PyTorch 类似，但支持的算子存在差异，需要在进行模型迁移前找出飞桨缺失的算子。

飞桨算子由各种 Python/C++ API 组成，包括基础数据结构、数据处理、模型组网、 优化器、分布式等。下面以其中较为常见的 3 类 API 算子进行介绍。

- 数据框架算子，包括张量处理、设备、基本数据类型等，如`paddle.abs`、`paddle.int64`、paddle.CPUPlace 等。
- 数据处理算子，包括数据集定义与加载、数据采用、多进程数据读取器算子，如`paddle.io.Dataset`、`paddle.io.BatchSampler`、`paddle.io.DataLoader`等。
- 模型组网算子，包括网络构建中使用到的全连接、激活等算子，如`paddle.nn.Linear、``paddle.nn.ReLU`等。

更多关于飞桨 API 的介绍，请参考：[飞桨 API 文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/Overview_cn.html)。

**【统计缺失算子】**

统计缺失算子时，在代码库找到网络结构及实现训练功能的 Python 文件（名称一般为 train.py model.py 等等），在脚本文件中查找所有相关算子（含数据框架类、数据预处理、网络结构算子），通过 AI 映射表，查找对应的飞桨算子。例如`torch.nn.ReLU`对应飞桨算子为`paddle.nn.ReLU。`飞桨与 PyTorch 关于 API 的映射关系可以参考：[API 映射表](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/model_convert/pytorch_api_mapping_cn.html)。

若该网页未能找到对应的算子，则可继续在[飞桨 API](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/index_cn.html)中搜索算子名称。

如果依然没有对应的飞桨算子，则计入缺失，可以尝试通过其他 API 算子组合实现、自定义算子实现或者在[Paddle repo](https://github.com/PaddlePaddle/Paddle/issues/new/choose)中提出一个新的需求。

**【注意事项】**

针对相同功能的算子，飞桨的命名可能与 PyTorch 不同；同名算子参数与功能也可能与 PyTorch 有区别。如 [paddle.optimizer.lr.StepDecay](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/optimizer/lr/StepDecay_cn.html#stepdecay)与[torch.optim.lr_scheduler.StepLR](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.StepLR.html#torch.optim.lr_scheduler.StepLR) 。

**【缺失算子处理策略】**

如果遇到飞桨不包含的算子或者 API，例如：某些算法实现存在调用了外部算子，而且飞桨也不包含该算子实现；或者其他框架存在的 API 或者算子，但是飞桨中没有这些算子。

1. 尝试使用替代实现进行迁移，飞桨中可以通过其他 API 的组合的形式进行功能实现。
2. 尝试使用飞桨的自定义算子功能：参考[自定义算子](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/custom_op/index_cn.html#zidingyisuanzi)。此方式存在一定的代码开发量。
3. 考虑通过自定义算子方式，使用其他已有第三方算子库实现：参考[PaddleDetection 自定义算子编译文档](https://github.com/PaddlePaddle/PaddleDetection/blob/release%2F2.5/ppdet/ext_op/README.md)。
4. 如果缺失的功能和算子无法规避，或者组合算子性能较差，严重影响网络的训练和推理，欢迎给飞桨提[ISSUE](https://github.com/PaddlePaddle/Paddle/issues/new/choose)，列出飞桨不支持的实现，飞桨开发人员会根据优先级进行实现。

## 二、模型前向对齐

模型前向对齐是模型迁移的最核心部分。在这一部分工作中，需要保证迁移之后的模型结构与原始模型一致，当两个模型固定为相同的权重时，对于相同的输入，二者的输出结果相同（即“前向对齐”）。模型前向对齐主要包括以下步骤：

1. **网络结构代码转换**：将定义模型网络结构的 PyTorch 代码转换成飞桨代码。由于飞桨定义模型网络结构的方式与 PyTorch 非常相似，因此主要的工作在于 API 的转换（例如内置的模块、函数）。
2. **权重转换**：只有当两个模型的权重相同时，对于相同的输入数据，两个模型才能输出相同的结果（即“前向对齐”）。为了检查模型前向对齐，需要使飞桨模型具有与原始模型相同的权重，因此需要把原始模型的权重转换为飞桨格式的模型权重，供飞桨模型加载。
3. **模型前向对齐验证**：让转换前后的模型加载相同的权重，给两个模型输入相同的数据，利用差异核验工具检查两个模型的输出是否一致。若二者的差异小于一定的阈值，则迁移后模型的正确性得到了验证。

### 2.1 网络结构代码转换

本步骤将定义模型网络结构的 PyTorch 代码转换成飞桨代码，使得转换前后的模型网络结构相同。由于飞桨对模型网络结构的定义方式与 PyTorch 非常相似，因此本步骤的工作主要在于将原始代码中调用的 PyTorch API（即原始代码中 import 的 torch 包的类、函数，例如 `torch.nn` 中的模块及 `torch.nn.functional` 中的函数等）替换成相应的飞桨 API。需要注意的是，飞桨的部分 API 与 PyTorch 中对应的 API 在功能与参数上存在一定区别，转换时需要留意。更多详细内容请参见“[解读网络结构转换](https://ku.baidu-int.com/knowledge/HFVrC7hq1Q/yKeL8Lljko/QuZY-m-iji/_U88vYhMWELV8m)”。

**【基本流程】**

由于 PyTorch 的 API 和飞桨的 API 非常相似，可以参考 [PyTorch-飞桨 API 映射表](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/08_api_mapping/pytorch_api_mapping_cn.html)，组网部分代码直接进行手动转换即可。

**【实战】**

BERT 网络结构的 PyTorch 实现: [torch_bert](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/examples/torch_migration/pipeline/models/pt_bert.py)

对应转换后的 PaddlePaddle 实现: [paddle_bert](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/examples/torch_migration/pipeline/models/pd_bert.py)

**【FAQ】**

Q：有什么其他没有在映射表中的 PyTorch API 是可以用飞桨中 API 实现的呢？

A：有的，例如：

- `torch.masked_fill`函数的功能目前可以使用`paddle.where`进行实现，可以参考[链接](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/faq/train_cn.html#paddletorch-masked-fillapi)。
- `pack_padded_sequence`和`pad_packed_sequence`这两个 API 目前飞桨中没有实现，可以直接在 RNN 或者 LSTM 的输入中传入`sequence_length`来实现等价的功能（可参考[ISSUE](https://github.com/PaddlePaddle/Paddle/issues/36882)）。



### 2.2 权重转换

只有当两个模型的权重相同时，对于相同的输入数据，两个模型才能输出相同的结果（即“前向对齐”）。Paddle 框架保存的模型文件名一般后缀为 `'.pdparams'` ，PyTorch 框架保存的模型文件名一般后缀为 `'.pt'`、`'.pth'` 或者 `'.bin'` 。因此需要把原始模型的权重转换为飞桨格式的模型权重，供飞桨模型加载。更多详细内容请参加“[模型权重转换详解](https://ku.baidu-int.com/knowledge/HFVrC7hq1Q/yKeL8Lljko/QuZY-m-iji/1x5E1jKmodoa3F)”。

**【基本流程】**

组网代码转换完成之后，需要对模型权重进行转换。权重的转换分为两个步骤：

1. **获取 PyTorch 权重**：下载 PyTorch 的模型权重，或随机生成并保存 PyTorch 权重
2. **转换为飞桨权重**：对二者存在差异的参数进行转换，并保存为飞桨的权重字典

下面以 BERT 模型为例介绍转换的过程：

- 如果 PyTorch repo 中已经提供权重，那么可以直接下载并进行后续的转换。huggingface 的 transformers 中提供了大部分模型参数，使用模型权重名称`model_name_or_path`即可加载（如`bert-base-uncased`）。或者从 huggingface 官网直接下载：https://huggingface.co/bert-base-uncased/tree/main
- 如果没有提供，则可以基于 PyTorch 代码，随机生成一个初始化权重(定义完 model 以后，使用`torch.save()` API 保存模型权重)，然后将生成的 PyTorch 权重 (`bert_sequence_classfy.pth`) 转换为飞桨模型权重。

```python
from transformers import BertModel
import torch

hf_model = BertModel.from_pretrained("bert-base-uncased")
hf_model.eval()
PATH = './torch_weight.bin'
torch.save(hf_model.state_dict(), PATH)
```

然后将生成的 PyTorch 权重 (`bert_sequence_classfy.pth`) 转换为飞桨模型权重，转换代码如下（代码解释详见代码后的 FAQ）

```python
def convert_pytorch_checkpoint_to_paddle(
    pytorch_checkpoint_path="pytorch_model.bin",
    paddle_dump_path="model_state.pdparams",
    version="old",
):
    do_not_transpose = []
    if version == "old":
        hf_to_paddle.update({
            "predictions.bias": "predictions.decoder_bias",
            ".gamma": ".weight",
            ".beta": ".bias",
        })
        do_not_transpose = do_not_transpose + ["predictions.decoder.weight"]

    pytorch_state_dict = torch.load(pytorch_checkpoint_path, map_location="cpu")
    paddle_state_dict = OrderedDict()
    for k, v in pytorch_state_dict.items():
        is_transpose = False
        if k[-7:] == ".weight":
            # embeddings.weight and LayerNorm.weight do not transpose
            if all(d not in k for d in do_not_transpose):
                if ".embeddings." not in k and ".LayerNorm." not in k:
                    if v.ndim == 2:
                        if 'embeddings' not in k:
                            v = v.transpose(0, 1)
                            is_transpose = True
                        is_transpose = False
        oldk = k
        print(f"Converting: {oldk} => {k} | is_transpose {is_transpose}")
        paddle_state_dict[k] = v.data.numpy()

    paddle.save(paddle_state_dict, paddle_dump_path)
```

**【FAQ】**

Q：权重转换过程中，PyTorch 和飞桨有哪些参数存在差异？

A：在权重转换的时候，需要注意`paddle.nn.Linear`以及`paddle.nn.LayerNorm`等 API 的权重保存格式和名称等与 PyTorch 稍有差异：

- `nn.Linear`层的 weight 参数：飞桨与 PyTorch 的参数存在互为转置的关系，因此在转换时需要进行转置，这可以参考上述的 torch2paddle 函数。有时还会遇到线性层被命名为 conv 的情况，但是我们依旧需要进行转置。

在权重转换的时候，不能只关注参数的名称，比如有些`paddle.nn.Linear`层，定义的变量名称为`conv`，这种情况下需要进行权重转置。权重转换时，建议同时打印飞桨和 PyTorch 对应权重的`shape`，以防止名称相似但是 `shape`不同的参数权重转换报错。

### 2.3 模型前向对齐验证

本步骤的目的是验证前面的网络结构代码转换的正确性，检查迁移前后模型结构是否一致。因此，需要让转换前后的模型加载相同的权重，给两个模型输入相同的数据，检查两个模型的输出是否一致。若二者的差异小于一定的阈值，则迁移后模型的正确性得到了验证。

**【基本流程】**

1. **PyTorch 前向传播**：定义 PyTorch 模型，加载权重，固定 seed，基于 numpy 生成随机数，转换为 `torch.Tensor`，送入网络，获取输出。
2. **飞桨前向传播**：定义飞桨模型，加载步骤 2.2 中转换后的权重，固定 seed，基于 numpy 生成随机数，转换为 `paddle.Tensor`，送入网络，获取输出。
3. **比较输出差异**：比较两个输出 tensor 的差值，如果小于阈值，即可完成自测。

**【实战】**

BERT 模型前向对齐验证可以参考如下示例代码：
https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/torch_migration/pipeline/Step1

**【核验】**

对于待复现的项目，前向对齐核验流程如下。

1. 准备输入：fake data

  - 使用参考代码的 dataloader，生成一个 batch 的数据，保存下来，在前向对齐时，直接从文件中读入。
  - 固定随机数种子，生成 numpy 随机矩阵，转化 tensor

1. 保存输出：

  - PaddlePaddle/PyTorch：dict，key 为 tensor 的 name（自定义），value 为 tensor 的值。最后将 dict 保存到文件中。建议命名为`forward_paddle.npy`和`forward_torch.npy`。

1. 自测：使用 reprod_log 加载 2 个文件，使用 report 功能，记录结果到日志文件中，建议命名为`forward_diff_log.txt`，观察 diff，二者 diff 小于特定的阈值即可。
2. 提交内容：新建文件夹，将`forward_paddle.npy`、`forward_torch.npy`与`forward_diff_log.txt`文件放在文件夹中，后续的输出结果和自查日志也放在该文件夹中，一并打包上传即可。
3. 注意：

- - PaddlePaddle 与 PyTorch 保存的 dict 的 key 需要保持相同，否则 report 过程可能会提示 key 无法对应，从而导致 report 失败，之后的`【核验】`环节也是如此。
  - 如果是固定随机数种子，建议将 fake data 保存到 dict 中，方便 check 参考代码和 PaddlePaddle 的输入是否一致。

**【注意事项】**

- 模型在前向对齐验证时，需要调用`model.eval()`方法，保证组网中的随机量被关闭，比如 Dropout 等。
- 给定相同的输入数据，为保证可复现性，如果有随机数生成，固定相关的随机种子。
- 网络结构对齐后，尽量使用训练好的预训练模型和真实的文本数据进行前向差值计算，使结果更准确。
- 在前向过程中，如果数据预处理过程运行出错，请先将 `paddle.io.DataLoader` 的 `num_workers` 参数设为 0，然后根据单个进程下的报错日志定位出具体的 bug。

**【FAQ】**

1. 在复杂的网络结构中，前向结果对不齐怎么办？

输出 diff 可以使用`np.mean(np.abs(o1 - o2))`进行计算，一般小于 1e-6 的话，可以认为前向没有问题。如果最终输出结果 diff 较大，可以用以下两种方法排查：

  - 可以按照模块排查问题，比如依次获取 backbone、neck、head 输出等，看下问题具体出现在哪个子模块，再进到子模块详细排查。
  - 如果最终输出结果差值较大，使用二分的方法进行排查，比如说 BERT，有 BertEmbeddings, 12 层的 Transformer 层 BertLayer（其中又有 BertAttention, BertIntermediate）以及最后的 BertPooler 层，那么完成模型组网和权重转换之后，如果模型输出没有对齐，可以尝试输出中间某一层输出的 tensor 进行对比，如果相同，则向后进行排查；如果不同，则继续向前进行排查，以此类推，直到找到导致没有对齐的操作。

1. 飞桨已经有了对于经典模型结构的实现，我还要重新实现一遍么？

这里建议自己根据 PyTorch 代码重新实现一遍，一方面是对整体的模型结构更加熟悉，另一方面也保证模型结构和权重完全对齐。

## 三、小数据集数据读取对齐

为快速验证数据读取以及后续的训练/评估/预测，可以准备一个小数据集（对句子二分类，其中包括 32 个句子以及他们对应的标签），数据位于 https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/torch_migration/pipeline/Step2/demo_sst2_sentence/demo.tsv

对于一个数据集，一般有以下一些信息需要重点关注：

  - 数据集名称、下载地址
  - 训练集/验证集/测试集
  - 数据集通用的预处理方法

**【基本流程】**

1. **Dataset 迁移**：使用`paddle.io.Dataset`完成数据集的单个样本读取。飞桨中数据集相关的 API 为`paddle.io.Dataset`，PyTorch 中对应为`torch.utils.data.Dataset`，二者功能一致，在绝大多数情况下，可以使用该类构建数据集。

此接口是描述 Dataset 方法和行为的抽象类，在具体实现的时候，需要继承这个基类，实现其中的`__getitem__`和`__len__`方法。

除了参考代码中相关实现，也可以参考待迁移项目中的说明。

1. **DataLoader 迁移**：迁移完 Dataset 之后，可以使用`paddle.io.DataLoader`，构建 Dataloader，对数据进行组 batch、批处理，送进网络进行计算。

`paddle.io.DataLoader`可以进行数据加载，将数据分成批数据，并提供加载过程中的采样。PyTorch 对应的实现为`torch.utils.data.DataLoader`，二者在功能上一致，仅在参数方面稍有差异：

- - 飞桨增加了`use_shared_memory`参数来选择是否使用共享内存加速数据加载过程。

**【实战】**

BERT 模型复现过程中，数据预处理和 Dataset、Dataloader 的检查可以参考该文件：https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/torch_migration/pipeline/Step2/test_data.py 使用方法可以参考[数据检查文档](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/torch_migration//pipeline/Step2/README.md)。

**【注意事项】**

- 如果代码中存在 random.random 或者其他 random 库的，可以先把随机数替换成固定数值，避免 random 对对齐结果产生影响。设置 seed 的方式当调用 random 库的顺序不同时产生的随机值也不同，可能导致对齐存在问题。
- 论文中一般会提供数据集的名称以及基本信息。复现过程中，我们在下载完数据之后，建议先检查下是否和论文中描述一致，否则可能存在的问题有：
- 数据集版本不同，比如论文中使用了 cnn_dailymail 的 v3.0.0 版本数据集，但是我们下载的是 cnn_dailymail 的 v1.0.0 版本数据集，如果不对其进行检查，可能会导致我们最终训练的数据量等与论文中有 diff
- 数据集使用方式不同，有些论文中，可能只是抽取了该数据集的子集进行方法验证，此时需要注意抽取方法，需要保证抽取出的子集完全相同。
- 在评估指标对齐时，我们可以固定 batch size，关闭 Dataloader 的 shuffle 操作。

此外，

- 有些自定义的数据处理方法，如果不涉及到深度学习框架的部分，可以直接复用。
- 对于特定任务中的数据预处理方法，比如说 Tokenizer，如果没有现成的 API 可以调用，可以参考 PaddleNLP 套件中的一些实现方法，比如`BertTokenizer`, `XLNetTokenizer`等。

**【FAQ】**

Q：还有其他导致不能对齐的因素么？

A：可以能存在以下因素：

  - 预处理方法顺序不一致：预处理的方法相同，顺序不同，比如先 padding 再做归一化与先做归一化再 padding，得到的结果可能是不同的。
  - 没有关闭 shuffle：在评估指标对齐时，需要固定 batch size，关闭 Dataloader 的 shuffle 操作。

## 四、评估指标对齐

评估指标是模型精度的度量。在自然语言处理中，不同任务使用的评估指标有所不同，为了检验迁移后的模型能否达到原模型的精度指标，需要保证使用的评估指标与原始代码一致以便对照。在这一步骤中，需要将原始代码中的评价指标替换成飞桨的等价实现。在前面的步骤中，已经检查了两个模型已前向对齐。因此，在此基础上让两个模型固定相同的权重，给两个模型输入相同的数据，分别计算评估指标，若二者评估指标结果相同，则说明评估指标对齐完成。

飞桨提供了一系列 Metric 计算类，比如说`Accuracy`, `Auc`, `Precision`, `Recall`等，而 PyTorch 中，目前可以通过组合的方式实现评估指标的计算，或者调用 [torchmetrics](https://torchmetrics.readthedocs.io/en/latest/)。在迁移过程中，需要注意保证对于该模块，给定相同的输入，二者输出完全一致。

**【基本流程】**

1. **获取 PyTorch 模型评估结果**：定义 PyTorch 模型，加载训练好的权重，获取评估结果。
2. **获取飞桨模型评估结果**：定义飞桨模型，加载训练好的权重（需要是从 PyTorch 转换得到），获取评估结果。
3. **比较结果差异**：使用 reprod_log 比较两个结果的差值，如果小于阈值，即可完成自测。

**【实战】**

评估指标对齐检查方法脚本：[test_metric.py](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/torch_migration/pipeline/Step2/test_metric.py)

**【核验】**

对于待迁移的项目，评估指标对齐核验流程如下。

1. 输入：dataloader, model
2. 输出：

 - 飞桨/PyTorch：dict，key 为 tensor 的 name（自定义），value 为具体评估指标的值。最后将 dict 使用 reprod_log 保存到各自的文件中，建议命名为`metric_paddle.npy`和`metric_pytorch.npy`。
  - 自测：使用 reprod_log 加载 2 个文件，使用 report 功能，记录结果到日志文件中，建议命名为`metric_diff_log.txt`，观察 diff，二者 diff 小于特定的阈值即可。

1. 提交内容：将`metric_paddle.npy`、`metric_pytorch.npy`与`metric_diff_log.txt`文件备份到`result`和`result/log`新建的文件夹中，后续的输出结果和自查日志也对应放在文件夹中，一并打包上传即可。
2. 注意：

  - 这部分需要使用真实数据
  - 需要检查 PyTorch 项目是否只是抽取了验证集/测试集中的部分文件。如果是的话，则需要保证飞桨和参考代码中 dataset 使用的数据集一致。

**【FAQ】**

Q：有哪些会导致评估出现精度偏差呢？

A：比如：

- - 使用 dataloader 参数没有保持一致：例如需要检查飞桨代码与 PyTorch 代码在`paddle.io.DataLoader` 的 `drop_last` 参数上是否保持一致 (文档[链接](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/io/DataLoader_cn.html#dataloader)，如果不一致，最后不够 batch-size 的数据可能不会参与评估，导致评估结果会有 diff。
  - 转换到不同设备上运行代码：在识别或者检索过程中，为了加速评估过程，往往会将评估函数由 CPU 实现改为 GPU 实现，由此会带来评估函数输出的不一致。这是由于 sort 函数对于相同值的排序结果不同带来的。在迁移的过程中，如果可以接受轻微的指标不稳定，可以使用飞桨的 sort 函数，如果对于指标非常敏感，同时对速度性能要求很高，可以给飞桨提[ISSUE](https://github.com/PaddlePaddle/Paddle/issues/new/choose)，由研发人员高优开发。
  - 评估参数和训练参数不一致：在检测任务中，评估流程往往和训练流程有一定差异，例如 RPN 阶段 NMS 的参数等，这里需要仔细检查评估时的超参数，不要将训练超参和评估超参弄混淆。
  - 评估时数据过滤规则不一致：在 OCR 等任务中，需要注意评估过程也会对 GT 信息进行修正，比如大小写等，也会过滤掉一些样本，这里需要注意过滤规则，确保有效评估数据集一致。

## 五、损失函数对齐

损失函数是训练模型时的优化目标，使用的损失函数会影响模型的精度。在模型迁移时，需要保证迁移后模型训练时使用的损失函数与原始代码中使用的损失函数一致，以便二者对照。在前面的步骤中，已经检查了两个模型已前向对齐。因此，在此基础上让两个模型固定相同的权重，给两个模型输入相同的数据，分别计算损失函数，若二者 loss 相同，则说明损失函数对齐完成。

飞桨与 PyTorch 均提供了很多损失函数，用于模型训练，具体的 API 映射表可以参考：[Loss 类 API 映射列表](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/model_convert/pytorch_api_mapping_cn.html#lossapi)。

**【基本流程】**

1. **计算 PyTorch 损失**：定义 PyTorch 模型，加载权重，加载 fake data 和 fake label（或者固定 seed，基于 numpy 生成随机数），转换为 `torch.Tensor`，送入网络，获取 loss 结果。
2. **计算飞桨损失**：定义飞桨模型，加载 fake data 和 fake label（或者固定 seed，基于 numpy 生成随机数），转换为 `paddle.Tensor`，送入网络，获取 loss 结果。
3. **比较结果差异**：使用 reprod_log 比较两个结果的差值，如果小于阈值，即可完成自测。

**【差异分析】**

以 CrossEntropyLoss 为例，主要区别为：

飞桨提供了对软标签、指定 softmax 计算维度的支持。如果 Pytorch 使用的 loss function 没有指定的 API，则可以尝试通过组合 API 的方式，实现自定义的 loss function。

**【实战】**

该部分可以参考[Step3/readme](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/torch_migration//pipeline/Step3#readme)

使用该文件生成 pytorch 损失函数:[Step3/torch_loss.py](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/torch_migration//pipeline/Step3/torch_loss.py)

使用该文件生成 paddle 损失函数:[Step3/paddle_loss.py](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/torch_migration//pipeline/Step3/paddle_loss.py)

**【核验】**

对于待迁移的项目，损失函数对齐核验流程如下。

1. 输入：fake data & label
2. 输出：飞桨/PyTorch：dict，key 为 tensor 的 name（自定义），value 为具体评估指标的值。最后将 dict 使用 reprod_log 保存到各自的文件中，建议命名为`loss_paddle.npy`和`loss_pytorch.npy`。
3. 自测：使用 reprod_log 加载 2 个文件，使用 report 功能，记录结果到日志文件中，建议命名为`loss_diff_log.txt`，观察 diff，二者 diff 小于特定的阈值即可。

**【注意事项】**

- 计算 loss 的时候，建议设置`model.eval()`，避免模型中随机量的问题。
- 飞桨目前支持在 `optimizer` 中通过设置 `params_groups` 的方式设置不同参数的更新方式，可以参考[代码示例](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/optimizer/optimizer.py#L140) 。
- 某些模型训练时，会使用梯度累加策略，即累加到一定 step 数量之后才进行参数更新，这时在实现上需要注意对齐。

**【FAQ】**

Q：为什么`nn.CrossEntropyLoss`出现不能对齐问题？

A：`paddle.nn.CrossEntropyLoss` 默认是在最后一维(axis=-1)计算损失函数，而 `torch.nn.CrossEntropyLoss` 是在 axis=1 的地方计算损失函数，因此如果输入的维度大于 2，这里需要保证计算的维(axis)相同，否则可能会出错。

## 六、模型训练超参对齐

模型的训练超参包括学习率、优化器、正则化策略等。这些超参数指定了模型训练过程中网络参数的更新方式，训练超参数的设置会影响到模型的收敛速度及收敛精度。在模型迁移时，需要保证迁移前后模型使用的训练超参数一致，以便对照二者的收敛情况。

**【基本流程】**

本部分对齐建议对照飞桨[优化器 API 文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/optimizer/Overview_cn.html)、[飞桨正则化 API 文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/regularizer/L2Decay_cn.html)与参考代码的优化器实现进行对齐，用之后的反向对齐统一验证该模块的正确性。

- **优化器**：飞桨中的 optimizer 有`paddle.optimizer`等一系列实现，PyTorch 中则有`torch.optim`等一系列实现。需要仔细对照飞桨与参考代码的优化器参数实现，确保优化器参数严格对齐。
- **学习率策略**：主要用于指定训练过程中的学习率变化曲线，这里可以将定义好的学习率策略，不断 step，即可得到对应的学习率值。对于迁移代码和 PyTorch 代码，学习率在整个训练过程中在相同的轮数相同的 iter 下应该保持一致，可以通过打印学习率值或者可视化二者学习率的 log 来查看差值。可以参考学习率对齐代码[learning.py](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/torch_migration//pipeline/Step4/test_lr_scheduler.py)。
- **正则化策略**：在模型训练中，L2 正则化可以防止模型对训练数据过拟合，L1 正则化可以用于得到稀疏化的权重矩阵。飞桨中有`paddle.regularizer.L1Decay`与`paddle.regularizer.L2Decay` API。PyTorch 中，torch.optim 集成的优化器只有 L2 正则化方法，直接在构建 optimizer 的时候，传入`weight_decay`参数即可。在如 Transformer 或者少部分 CNN 模型中，存在一些参数不做正则化(正则化系数为 0)的情况。这里需要找到这些参数并对齐取消实施正则化策略，可以参考[这里](https://github.com/PaddlePaddle/PaddleClas/blob/f677f61ea3b34281968d1059f4f8f13c1d787e67/ppcls/arch/backbone/model_zoo/resnest.py#L74)，对特定参数进行修改。

**【区别】**

以 SGD 等优化器为例，飞桨与 Pytorch 的优化器区别主要如下：

- 飞桨中，需要首先构建学习率策略，再传入优化器对象中；对于 PyTorch，如果希望使用更丰富的学习率策略，需要先构建优化器，再传入学习率策略类 API。
- 飞桨在优化器中增加了对梯度裁剪的支持，在训练 GAN、NLP、多模态等任务时较为常用。
- 飞桨的 SGD 不支持动量更新、动量衰减和 Nesterov 动量。若需实现这些功能，请使用`paddle.optimizer.Momentum` API。
- 飞桨的 optimizer 中支持 L1Decay/L2Decay。

**【FAQ】**

1. 怎么实现参数分组学习率策略？

飞桨目前支持在 `optimizer` 中通过设置 `params_groups` 的方式设置不同参数的更新方式，可以参考[代码示例](https://github.com/PaddlePaddle/Paddle/blob/166ff39a20f39ed590a0cf868c2ad2f15cf0bbb1/python/paddle/optimizer/optimizer.py#L138) 。

1. 有没有什么其他影响优化不对齐的原因？

  - 有些模型训练时，会使用梯度累加策略，即累加到一定 step 数量之后才进行参数更新，这时在实现上需要注意对齐。
  - 在文本分类领域，大多数 Transformer 模型都采用了 AdamW 优化器，并且会设置 weigh decay，同时部分参数设置为 no weight decay，例如位置编码的参数通常设置为 no weight decay，no weight decay 参数设置不正确，最终会有明显的精度损失，需要特别注意。一般可以通过分析模型权重来发现该问题，分别计算官方模型和复现模型每层参数权重的平均值、方差，对每一层依次对比，有显著差异的层可能存在问题，因为在 weight decay 的作用下，参数权重数值会相对较小，而未正确设置 no weight decay，则会造成该层参数权重数值异常偏小设置 no weight decay 可以参照[这里](https://github.com/PaddlePaddle/PaddleClas/blob/release%2F2.3/ppcls/arch/backbone/model_zoo/resnest.py#L72)。
  - 在 OCR 识别等任务中，`Adadelta`优化器常常被使用，该优化器与 PyTorch 实现目前稍有不同，但是不影响模型训练精度对齐。在做前反向对齐时，可以将该优化器替换为 Adam 等优化器（飞桨与参考代码均需要替换）；对齐完成之后，再使用`Adadelta`优化器进行完整的训练精度对齐。

1. 有没有什么任务不需要进行优化对齐的呢？

在某些任务中，比如说深度学习可视化、可解释性等任务中，一般只要求模型前向过程，不需要训练，此时优化器、学习率等用于模型训练的模块对于该类模型迁移是不需要的。

1. 飞桨的学习率策略对不齐怎么办？

  - 飞桨中参数的学习率受到优化器学习率和`ParamAttr`中设置的学习率影响，在对齐时也需要保证这个部分的行为一致。
  - 有些网络的学习率策略比较细致，比如带 warmup 的学习率策略，这里需要保证起始学习率等参数都完全一致。

## 七、反向梯度对齐

反向梯度对齐的目的是确保迁移后的模型反向传播以及权重更新的行为与原始模型一致，同时也是对上一步*模型训练超参对齐*的验证。具体的检验方法是通过两次（或以上）迭代训练进行检查（这是因为第一次迭代之后，会根据反向传播计算得到的梯度与设定的训练超参更新模型参数，第二次迭代前向传播时，使用的是更新后的参数），若迁移前后的模型第二个迭代的训练 loss 一致，说明二者更新后的参数一致，则可以认为二者反向已对齐。

**【基本流程】**

此处可以通过 numpy 生成 fake data 和 label（推荐），也可以准备固定的真实数据。具体流程如下：

1. **检查训练超参**：检查两个代码的训练超参数全部一致，如优化器及其超参数、学习率、BatchNorm/LayerNorm 中的 eps 等。
2. **关闭随机操作**：将飞桨与 PyTorch 网络中涉及的所有随机操作全部关闭，如 dropout、drop_path 等，推荐将模型设置为 eval 模式（`model.eval()`）。
3. **训练并比较损失**：加载相同的 weight dict（可以通过 PyTorch 来存储随机的权重），将准备好的数据分别传入网络并迭代，观察二者 loss 是否一致（此处 batch-size 要一致，如果使用多个真实数据，要保证传入网络的顺序一致）。如果经过 2 次迭代以上，loss 均可以对齐，则基本可以认为反向对齐。

**【实战】**

本部分可以参考文档：[反向对齐操作文档](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/torch_migration/pipeline/Step4/README.md#反向对齐操作方法)。

**【核验】**

对于待迁移的项目，反向对齐核验流程如下。

1. 输入：fake data & label
2. 输出：

  - 飞桨/PyTorch：dict，key 为 tensor 的 name（自定义），value 为具体 loss 的值。最后将 dict 使用 reprod_log 保存到各自的文件中，建议命名为`losses_paddle.npy`和`losses_pytorch.npy`。

1. 自测：使用 reprod_log 加载 2 个文件，使用 report 功能，记录结果到日志文件中，建议命名为`losses_diff_log.txt`，观察 diff，二者 diff 小于特定的阈值即可。
2. 注意：

  - loss 需要保存至少 2 轮以上。
  - 在迭代的过程中，需要保证模型的 batch size 等超参数完全相同
  - 在迭代的过程中，需要设置`model.eval()`，使用固定的假数据，同时加载相同权重的预训练模型。

**【FAQ】**

1. 怎么打印反向梯度？

 - 飞桨打印反向和参数更新，可以参考[代码实例](https://github.com/jerrywgz/PaddleDetection/blob/63783c25ca12c8a7e1d4d5051d0888b64588e43c/ppdet/modeling/backbones/resnet.py#L598)。
  - PyTorch 打印反向和参数更新，可以参考[代码实例](https://github.com/jerrywgz/mmdetection/blob/ca9b8ef3e3770c4ad268a2fad6c55eb5d066e1b4/mmdet/models/backbones/resnet.py#L655)。

1. 反向没有对齐怎么办？

反向对齐时，如果第一轮 loss 就没有对齐，则需要仔细先排查模型前向部分。

如果第二轮开始，loss 开始无法对齐，则首先需要排查下超参数的差异，没问题的话，在`loss.backward()`方法之后，使用`tensor.grad`获取梯度值，二分的方法查找 diff，定位出飞桨与 PyTorch 梯度无法对齐的 API 或者操作，然后进一步验证。

梯度的打印方法示例代码如下所示，注释掉的内容即为打印网络中所有参数的梯度 shape。

```python
# 代码地址：https://github.com/PaddlePaddle/PaddleNLP/blob/develop/examples/torch_migration/pipeline/Step4/test_bp.py
    def pd_train_some_iters(model,
                        criterion,
                        optimizer,
                        fake_data,
                        fake_label,
                        max_iter=2):
        paddle_dump_path = '../weights/paddle_weight.pdparams'
        config = PDBertConfig()
        model = PDBertForSequenceClassification(config)
        checkpoint = paddle.load(paddle_dump_path)
        model.bert.load_dict(checkpoint)

        classifier_weights = paddle.load(
            "../classifier_weights/paddle_classifier_weights.bin")
        model.load_dict(classifier_weights)
        model.eval()
        criterion = paddle.nn.CrossEntropy()
        decay_params = [
            p.name for n, p in model.named_parameters()
            if not any(nd in n for nd in ["bias", "norm"])
        ]
        optimizer = paddle.optimizer.AdamW(
            learning_rate=3e-5,
            parameters=model.parameters(),
            weight_decay=1e-2,
            epsilon=1e-6,
            apply_decay_param_fun=lambda x: x in decay_params,
        )
        loss_list = []
        for idx in range(max_iter):
            input_ids = paddle.to_tensor(fake_data)
            labels = paddle.to_tensor(fake_label)

            output = model(input_ids)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()
            loss_list.append(loss)
        return loss_list
```



如果只希望打印特定参数的梯度，可以用下面的方式。

```plain
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

**【基本流程】**

该部分内容与 “三、小数据集数据读取对齐” 内容基本一致，也可使用 PaddleNLP[内置数据集](https://paddlenlp.readthedocs.io/zh/latest/data_prepare/dataset_load.html)，参考 PyTorch 的代码，实现训练集数据读取与预处理模块即可。

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

1. 数据读取无法对齐怎么办？

  - 如果是全量数据使用，对结果不会有影响，如果是按照比例选取子集进行训练，则建议重新根据参考代码实现数据读取部分，保证子集完全一致。
  - 如果数据处理过程中涉及到随机数生成，建议固定 seed (`np.random.seed(0)`, `random.seed(0)`)，查看迁移代码和参考代码处理后的数据是否有 diff。
  - 对文本进行 tokenizer 处理时，需要确定文本的截断策略，padding 策略。

## 九、网络初始化对齐

本部分对齐建议对照飞桨初始化 API 文档与参考代码的初始化实现对齐。

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

- 对于不同的深度学习框架，网络初始化在大多情况下，即使值的分布完全一致，也无法保证值完全一致，这里也是论文复现中不确定性比较大的地方。如果十分怀疑初始化导致的问题，建议将参考的初始化权重转成 paddle 模型，加载该初始化模型训练，看下收敛精度。
- CNN 对于模型初始化相对来说没有那么敏感，在迭代轮数与数据集足够的情况下，最终精度指标基本接近；而 transformer 系列模型对于初始化比较敏感，在 transformer 系列模型训练对齐过程中，建议对这一块进行重点检查。

2、如何对齐 torch.nn.init.constant_() ？

飞桨中目前没有 `torch.nn.init.constant_()`的方法，如果希望对参数赋值为常数，可以使用 `paddle.nn.initializer.Constant`API；或者可以参考下面的实现。更加具体的解释可以参考：[模型参数初始化对齐](https://github.com/PaddlePaddle/models/blob/release/2.3/tutorials/article-implementation/initializer.md)。

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
- Transformer、等模型对于初始化比较敏感，在这些模型训练对齐过程中，建议对这一块进行重点检查；

## 十、训练精度对齐

**【基本流程】**

完成前面的步骤之后，就可以开始全量数据的训练对齐任务了。按照下面的步骤进行训练精度对齐。

1. 准备 train/eval data, loader, model
2. 对 model 按照论文所述进行初始化(如果论文中提到加载了预训练模型，则按需加载 pretrained model)
3. 加载配置，开始训练，迭代得到最终模型与评估指标，将评估指标使用 reprod_log 保存到文件中。
4. 将 PaddlePaddle 提供的参考指标使用 reprod_log 提交到另一个文件中。
5. 使用 reprod_log 排查 diff，小于阈值，即可完成自测。

**【注意事项】**

- 【强烈】建议先做完反向对齐之后再进行模型训练对齐，二者之间的不确定量包括：数据集、PaddlePaddle 与参考代码在模型 training mode 下的区别，初始化参数。
- 在训练对齐过程中，受到较多随机量的影响，精度有少量 diff 是正常的，以 SST-2 数据集的分类为例，diff 在 0.15%以内可以认为是正常的，这里可以根据不同的任务，适当调整对齐检查的阈值(`ReprodDiffHelper.report`函数中的`diff_threshold`参数)。
- 训练过程中的波动是正常的，如果最终收敛结果不一致，可以
  - 仔细排查 Dropout、BatchNorm 以及其他组网模块及超参是否无误。
  - 基于参考代码随机生成一份预训练模型，转化为 PaddlePaddle 的模型，并使用 PaddlePaddle 加载训练，对比二者的收敛曲线与最终结果，排查初始化影响。
  - 使用参考代码的 Dataloader 生成的数据，进行模型训练，排查 train dataloader 的影响。


**【实战】**

本部分可以参考文档：[训练对齐操作文档](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/examples/torch_migration/pipeline/Step5/README.md)。

**【核验】**

对于待复现的项目，训练对齐核验流程如下。

1. 输入：train/eval dataloader, model
2. 输出：

  - PaddlePaddle：dict，key 为保存值的 name（自定义），value 为具体评估指标的值。最后将 dict 使用 reprod_log 保存到文件中，建议命名为`train_align_paddle.npy`。
  - benchmark：dict，key 为保存值的 name（自定义），value 为模型迁移的评估指标要求的值。最后将 dict 使用 reprod_log 保存到文件中，建议命名为`train_align_benchmark.npy`。

自测：使用 reprod_log 加载 2 个文件，使用 report 功能，记录结果到日志文件中，建议命名为`train_align_diff_log.txt`，观察 diff，二者 diff 小于特定的阈值即可。

**【FAQ】**

1. 训练过程怎么更好地对齐呢？

  - 有条件的话，迁移工作之前最好先基于 PyTorch 代码完成训练，保证与 PyTorch 的代码训练精度符合预期，并且将训练策略和训练过程中的关键指标记录保存下来，比如每个 epoch 的学习率、Train Loss、Eval Loss、Eval Acc 等，在迁移网络的训练过程中，将关键指标保存下来，这样可以将 Paddle 与 PyTorch 训练中关键指标的变化曲线绘制出来，能够很方便地进行对比；
  - 如果训练较大的数据集，1 次完整训练的成本比较高，此时可以隔一段时间查看一下，如果精度差异比较大，建议先停掉实验，排查原因。

1. 如果训练过程中出现不收敛的情况，怎么办？

  - 简化网络和数据，实验是否收敛；
  - 如果是基于原有实现进行改动，可以尝试控制变量法，每次做一个改动，逐个排查；
  - 检查学习率是否过大、优化器设置是否合理，排查下 weight decay 是否设置正确；
  - 保存不同 step 之间的模型参数，观察模型参数是否更新。

1. 如果训练的过程中出 nan 怎么办？一般是因为除 0 或者 log0 的情况， 可以着重看下几个部分：
   1. 如果有预训练模型的话，可以确认下是否加载正确
   2. 确认下 reader 的预处理中是否会出现空的 tensor（检测任务中一般会出现该问题）
   3. 模型结构中计算 loss 的部分是否有考虑到正样本为 0 的情况
   4. 该问题也可能是某个 API 的数值越界导致的，可以测试较小的输入是否还会出现 nan。
2. 其他细分场景下有什么导致训练不对齐的原因？

  - 小数据上指标波动可能比较大，时间允许的话，可以跑多次实验，取平均值。
  - Transformer 系列模型，在模型量级比较小的情况下，使用更大的 batch size 以及对应的学习率进行训练往往会获得更高的精度，在迁移时，建议保证 batch size 和学习率完全一致，否则即使精度对齐了，也可能会隐藏其他没有对齐的风险项。
  - 目标检测、图像分割等任务中，训练通常需要加载 backbone 的权重作为预训练模型，注意在训练对齐时，需要加载转换过来的 backbone 权重。
  - 生成任务中，训练时经常需要固定一部分网络参数。对于一个参数`param`，可以通过`param.trainable = False`来固定。
  - 在训练 GAN 时，通常通过 GAN 的 loss 较难判断出训练是否收敛，建议每训练几次迭代保存一下训练生成的图像，通过可视化判断训练是否收敛。
  - 在训练 GAN 时，如果飞桨实现的代码已经可以与参考代码完全一致，参考代码和飞桨代码均难以收敛，则可以在训练的时候，可以判断一下 loss，如果 loss 大于一个阈值或者直接为 NAN，说明训崩了，就终止训练，使用最新存的参数重新继续训练。可以参考该链接的实现：[链接](https://github.com/JennyVanessa/Paddle-GI)。

1. 怎样设置运行设备?

对于飞桨来说，通过`paddle.set_device`函数（全局，会决定 Dataloader、模型组网、to_tensor 等操作产出的数据所在的设备）来确定模型结构是运行在什么设备上，对于 torch 来说，则是通过`model.to("device")` （局部，这句话仅影响 model 所在的设备）来确定模型结构的运行设备。

## 十一、模型预测验证

**【基本流程】**

模型训练完成之后，使用该模型基于训练引擎进行预测，主要包含：

1. 定义模型结构，加载模型权重。
2. 加载文本，对其进行数据预处理。
3. 模型预测。
4. 对模型输出进行后处理，获取最终输出结果。

**【注意事项】**

在模型评估过程中，为了保证数据可以组成图片尺寸相同的 batch，一般会使用 resize/padding 等方法以保持尺度的一致性。

**【实战】**

主要源代码如下所示：

```python
@paddle.no_grad()
def main():
    # 模型定义
    paddle_dump_path = '../weights/paddle_weight.pdparams'
    config = BertConfig()
    model = BertForSequenceClassification(config)
    checkpoint = paddle.load(paddle_dump_path)
    model.bert.load_dict(checkpoint)

    classifier_weights = paddle.load(
        "../classifier_weights/paddle_classifier_weights.bin")
    model.load_dict(classifier_weights)

    model.eval()
    tokenizer = PPNLPBertTokenizer.from_pretrained("bert-base-uncased")
    # 要预测的句子
    data = get_data()
    softmax = nn.Softmax()
    # 预测的各类别的概率值
    output = softmax(model(**data)[0]).numpy()

    # 概率值最大的类别
    class_id = output.argmax()
    # 对应的概率值
    prob = output[0][class_id]
    print(f"class_id: {class_id}, prob: {prob}")
    return output
```

完整代码地址：[predict.py](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/examples/torch_migration/pipeline/Step2/predict.py)

**【核验】**

预测程序按照文档中的命令操作可以正常跑通，终端输出结果。

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

单机多卡的代码实现请参考：[链接](https://github.com/PaddlePaddle/models/blob/release%2F2.3/tutorials/mobilenetv3_prod/Step6/train.py)，单机单卡属于单机多卡的特殊形式（卡数为 1），这里也可以用于单机单卡的训练过程。

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

对于单机单卡或者单机多卡的启动脚本可以参考：https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/language_model/bert

对于单机单卡，启动脚本如下所示

```shell
unset CUDA_VISIBLE_DEVICES
python -m paddle.distributed.launch --gpus "0" run_glue.py \
    --model_type bert \
    --model_name_or_path bert-base-uncased \
    --task_name SST-2 \
    --max_seq_length 128 \
    --batch_size 32   \
    --learning_rate 2e-5 \
    --num_train_epochs 3 \
    --logging_steps 1 \
    --save_steps 500 \
    --output_dir ./tmp/ \
    --device gpu \
    --use_amp False
```

对于单机多卡（示例中为 4 卡训练），启动脚本如下所示。

```shell
unset CUDA_VISIBLE_DEVICES
python -m paddle.distributed.launch --gpus "0,1,2,3" run_glue.py \
    --model_type bert \
    --model_name_or_path bert-base-uncased \
    --task_name SST-2 \
    --max_seq_length 128 \
    --batch_size 32   \
    --learning_rate 2e-5 \
    --num_train_epochs 3 \
    --logging_steps 1 \
    --save_steps 500 \
    --output_dir ./tmp/ \
    --device gpu \
    --use_amp False
```

注意：这里 4 卡训练时，虽然单卡的 batch size 没有变化(32)，但是总卡的 batch size 相当于是单卡的 4 倍，因此学习率也设置为了单卡时的 4 倍。

**【实战】**

本部分可以参考 paddlenlp 库中的例子：[单机多卡训练](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/language_model/bert)。

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


<p align="center">
  <img src="https://raw.githubusercontent.com/ymyjl/docs/torch_migrate/docs/guides/model_convert/pictures/information.png" align="middle"  width="500" />
</p>

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
