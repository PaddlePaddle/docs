# VisualDL 工具简介


<p align="center">
  <img src="http://visualdl.bj.bcebos.com/images/vdl-logo.png" width="70%"/>
</p>



VisualDL 是飞桨可视化分析工具，以丰富的图表呈现训练参数变化趋势、模型结构、数据样本、直方图、PR 曲线及高维数据分布。可帮助用户更清晰直观地理解深度学习模型训练过程及模型结构，进而实现高效的模型优化。

具体功能使用方式请参见**VisualDL 使用指南**。项目正处于高速迭代中，敬请期待新组件的加入。

VisualDL 支持浏览器种类：Chrome（81 和 83）、Safari 13、Firefox（77 和 78）、Edge（Chromium 版）。

VisualDL 原生支持 python 的使用， 通过在模型的 Python 配置中添加几行代码，便可为训练过程提供丰富的可视化支持。



## 目录

* [核心亮点](#核心亮点)
* [安装方式](#安装方式)
* [使用方式](#使用方式)
* [可视化功能概览](#可视化功能概览)
* [开源贡献](#开源贡献)
* [更多细节](#更多细节)
* [技术交流](#技术交流)



## 核心亮点

### 简单易用

API 设计简洁易懂，使用简单。模型结构一键实现可视化。

### 功能丰富

功能覆盖标量、数据样本、图结构、直方图、PR 曲线及数据降维可视化。

### 高兼容性

全面支持 Paddle、ONNX、Caffe 等市面主流模型结构可视化，广泛支持各类用户进行可视化分析。

### 全面支持

与飞桨服务平台及工具组件全面打通，为您在飞桨生态系统中提供最佳使用体验。



## 安装方式

### 使用 pip 安装

```shell
pip install --upgrade --pre visualdl
```

### 使用代码安装

```
git clone https://github.com/PaddlePaddle/VisualDL.git
cd VisualDL

python setup.py bdist_wheel
pip install --upgrade dist/visualdl-*.whl
```

需要注意，官方自 2020 年 1 月 1 日起不再维护 Python2，为了保障代码可用性，VisualDL 现仅支持 Python3

## 使用方式

VisualDL 将训练过程中的数据、参数等信息储存至日志文件中后，启动面板即可查看可视化结果。

### 1. 记录日志

VisualDL 的后端提供了 Python SDK，可通过 LogWriter 定制一个日志记录器，接口如下：

```python
class LogWriter(logdir=None,
                comment='',
                max_queue=10,
                flush_secs=120,
                filename_suffix='',
                write_to_disk=True,
                **kwargs)
```

#### 接口参数

| 参数            | 格式    | 含义                                                         |
| --------------- | ------- | ------------------------------------------------------------ |
| logdir          | string  | 日志文件所在的路径，VisualDL 将在此路径下建立日志文件并进行记录，如果不填则默认为`runs/${CURRENT_TIME}` |
| comment         | string  | 为日志文件夹名添加后缀，如果制定了 logdir 则此项无效           |
| max_queue       | int     | 日志记录消息队列的最大容量，达到此容量则立即写入到日志文件   |
| flush_secs      | int     | 日志记录消息队列的最大缓存时间，达到此时间则立即写入到日志文件 |
| filename_suffix | string  | 为默认的日志文件名添加后缀                                   |
| write_to_disk   | boolean | 是否写入到磁盘                                               |

#### 示例

设置日志文件并记录标量数据：

```python
from visualdl import LogWriter

# 在`./log/scalar_test/train`路径下建立日志文件
with LogWriter(logdir="./log/scalar_test/train") as writer:
    # 使用 scalar 组件记录一个标量数据
    writer.add_scalar(tag="acc", step=1, value=0.5678)
    writer.add_scalar(tag="acc", step=2, value=0.6878)
    writer.add_scalar(tag="acc", step=3, value=0.9878)
```

### 2. 启动面板

在上述示例中，日志已记录三组标量数据，现可启动 VisualDL 面板查看日志的可视化结果，共有两种启动方式：

#### 在命令行启动

使用命令行启动 VisualDL 面板，命令格式如下：

```python
visualdl --logdir <dir_1, dir_2, ... , dir_n> --host <host> --port <port> --cache-timeout <cache_timeout> --language <language> --public-path <public_path> --api-only
```

参数详情：

| 参数            | 意义                                                         |
| --------------- | ------------------------------------------------------------ |
| --logdir        | 设定日志所在目录，可以指定多个目录，VisualDL 将遍历并且迭代寻找指定目录的子目录，将所有实验结果进行可视化 |
| --model         | 设定模型文件路径(非文件夹路径)，VisualDL 将在此路径指定的模型文件进行可视化，目前可支持 PaddlePaddle、ONNX、Keras、Core ML、Caffe 等多种模型结构，详情可查看[graph 支持模型种类]([https://github.com/PaddlePaddle/VisualDL/blob/develop/docs/components/README.md#Graph--%E7%BD%91%E7%BB%9C%E7%BB%93%E6%9E%84%E7%BB%84%E4%BB%B6](https://github.com/PaddlePaddle/VisualDL/blob/develop/docs/components/README.md#Graph--网络结构组件)) |
| --host          | 设定 IP，默认为`127.0.0.1`                                    |
| --port          | 设定端口，默认为`8040`                                       |
| --cache-timeout | 后端缓存时间，在缓存时间内前端多次请求同一 url，返回的数据从缓存中获取，默认为 20 秒 |
| --language      | VisualDL 面板语言，可指定为'EN'或'ZH'，默认为浏览器使用语言   |
| --public-path   | VisualDL 面板 URL 路径，默认是'/app'，即访问地址为'http://&lt;host&gt;:&lt;port&gt;/app' |
| --api-only      | 是否只提供 API，如果设置此参数，则 VisualDL 不提供页面展示，只提供 API 服务，此时 API 地址为'http://&lt;host&gt;:&lt;port&gt;/&lt;public_path&gt;/api'；若没有设置 public_path 参数，则默认为'http://&lt;host&gt;:&lt;port&gt;/api' |

针对上一步生成的日志，启动命令为：

```
visualdl --logdir ./log
```

#### 在 Python 脚本中启动

支持在 Python 脚本中启动 VisualDL 面板，接口如下：

```python
visualdl.server.app.run(logdir,
                        host="127.0.0.1",
                        port=8080,
                        cache_timeout=20,
                        language=None,
                        public_path=None,
                        api_only=False,
                        open_browser=False)
```

请注意：除`logdir`外，其他参数均为不定参数，传递时请指明参数名。

接口参数具体如下：

| 参数          | 格式                                             | 含义                                                         |
| ------------- | ------------------------------------------------ | ------------------------------------------------------------ |
| logdir        | string 或 list[string_1, string_2, ... , string_n] | 日志文件所在的路径，VisualDL 将在此路径下递归搜索日志文件并进行可视化，可指定单个或多个路径 |
| model         | string                                           | 模型文件路径(非文件夹路径)，VisualDL 将在此路径指定的模型文件进行可视化 |
| host          | string                                           | 指定启动服务的 ip，默认为`127.0.0.1`                          |
| port          | int                                              | 启动服务端口，默认为`8040`                                   |
| cache_timeout | int                                              | 后端缓存时间，在缓存时间内前端多次请求同一 url，返回的数据从缓存中获取，默认为 20 秒 |
| language      | string                                           | VisualDL 面板语言，可指定为'en'或'zh'，默认为浏览器使用语言   |
| public_path   | string                                           | VisualDL 面板 URL 路径，默认是'/app'，即访问地址为'http://<host>:<port>/app' |
| api_only      | boolean                                          | 是否只提供 API，如果设置此参数，则 VisualDL 不提供页面展示，只提供 API 服务，此时 API 地址为'http://<host>:<port>/<public_path>/api'；若没有设置 public_path 参数，则默认为 http://<host>:<port>/api' |
| open_browser  | boolean                                          | 是否打开浏览器，设置为 True 则在启动后自动打开浏览器并访问 VisualDL 面板，若设置 api_only，则忽略此参数 |

针对上一步生成的日志，我们的启动脚本为：

```python
from visualdl.server import app

app.run(logdir="./log")
```

在使用任意一种方式启动 VisualDL 面板后，打开浏览器访问 VisualDL 面板，即可查看日志的可视化结果，如图：

<p align="center">
  <img src="https://user-images.githubusercontent.com/48054808/82786044-67ae9880-9e96-11ea-8a2b-3a0951a6ec19.png" width="60%"/>
</p>



## 可视化功能概览

### Scalar

以图表形式实时展示训练过程参数，如 loss、accuracy。让用户通过观察单组或多组训练参数变化，了解训练过程，加速模型调优。具有两大特点：

#### 动态展示

在启动 VisualDL 后，LogReader 将不断增量的读取日志中数据并供前端调用展示，因此能够在训练中同步观测指标变化，如下图：

<p align="center">
  <img src="http://visualdl.bj.bcebos.com/images/dynamic_display.gif" width="60%"/>
</p>



#### 多实验对比

只需在启动 VisualDL 时将每个实验日志所在路径同时传入即可，每个实验中相同 tag 的指标将绘制在一张图中同步呈现，如下图：

<p align="center">
  <img src="http://visualdl.bj.bcebos.com/images/multi_experiments.gif" width="100%"/>
</p>



### Image

实时展示训练过程中的图像数据，用于观察不同训练阶段的图像变化，进而深入了解训练过程及效果。

<p align="center">
<img src="http://visualdl.bj.bcebos.com/images/image-eye.gif" width="60%"/>
</p>



### Audio

实时查看训练过程中的音频数据，监控语音识别与合成等任务的训练过程。

<p align="center">
<img src="https://user-images.githubusercontent.com/48054808/89017647-38605000-d34d-11ea-9d75-7d10b9854c36.gif" width="100%"/>
</p>



### Graph

一键可视化模型的网络结构。可查看模型属性、节点信息、节点输入输出等，并支持节点搜索，辅助用户快速分析模型结构与了解数据流向。

<p align="center">
<img src="https://user-images.githubusercontent.com/48054808/84483052-5acdd980-accb-11ea-8519-1608da7ee698.png" width="100%"/>
</p>



### Histogram

以直方图形式展示 Tensor（weight、bias、gradient 等）数据在训练过程中的变化趋势。深入了解模型各层效果，帮助开发者精准调整模型结构。

- Offset 模式

<p align="center">
<img src="https://user-images.githubusercontent.com/48054808/86551031-86647c80-bf76-11ea-8ec2-8c86826c8137.png" width="100%"/>
</p>



- Overlay 模式

<p align="center">
<img src="https://user-images.githubusercontent.com/48054808/86551033-882e4000-bf76-11ea-8e6a-af954c662ced.png" width="100%"/>
</p>



### PR Curve

精度-召回率曲线，帮助开发者权衡模型精度和召回率之间的平衡，设定最佳阈值。

<p align="center">
<img src="https://user-images.githubusercontent.com/48054808/86738774-ee46c000-c067-11ea-90d2-a98aac445cca.png" width="100%"/>
</p>


### High Dimensional

将高维数据进行降维展示，目前支持 T-SNE、PCA 两种降维方式，用于深入分析高维数据间的关系，方便用户根据数据特征进行算法优化。

<p align="center">
<img src="http://visualdl.bj.bcebos.com/images/high_dimensional_test.png" width="100%"/>
</p>

## 开源贡献

VisualDL 是由 [PaddlePaddle](https://www.paddlepaddle.org/) 和 [ECharts](https://echarts.apache.org/) 合作推出的开源项目。
Graph 相关功能由 [Netron](https://github.com/lutzroeder/netron) 提供技术支持。
欢迎所有人使用，提意见以及贡献代码。


## 更多细节

想了解更多关于 VisualDL 可视化功能的使用详情介绍，请查看**VisualDL 使用指南**。

## 技术交流

欢迎您加入 VisualDL 官方 QQ 群：1045783368 与飞桨团队以及其他用户共同针对 VisualDL 进行讨论与交流。
