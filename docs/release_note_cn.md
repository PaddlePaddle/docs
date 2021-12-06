
# 2.2.1 Release Note

## 1. 重要更新

我们很高兴的发布飞桨框架2.2.1版本，主要是对2.2.0中一些功能和性能问题的修复，并对部分功能点做了增强，重点如下：

- 新增  ``paddle.linalg.triangular_solve``，用于计算带有三角系数矩阵的线性方程组。
- 新增 `paddle.device.cuda.graphs.CUDAGraph` API，支持NVIDIA的[CUDA Graph](https://developer.nvidia.com/blog/cuda-graphs/)功能，注意目前该API还处于实验阶段，尚未稳定。
- 修复了基础API、Tensor 索引中的已知问题。


## 2. 训练框架（含分布式）

### （1）新功能

#### API

- 新增``paddle.linalg.triangular_solve`` API，用于计算带有三角系数矩阵的线性方程组。([#36714](https://github.com/PaddlePaddle/Paddle/pull/36714))
- 新增`paddle.device.cuda.graphs.CUDAGraph` API，支持NVIDIA的[CUDA Graph](https://developer.nvidia.com/blog/cuda-graphs/)功能，可以将GPU计算全部捕捉到一张CUDA Graph中，往后多次调用，可以去除框架的额外开销，提升运行性能。注意目前该API还处于实验阶段，尚未稳定。([#37109](https://github.com/PaddlePaddle/Paddle/pull/37109))
- 新增``paddle.incubate.graph_send_recv`` API，主要应用于图学习领域，目的是为了减少在消息传递过程中带来的中间变量显存或内存的损耗，包含 SUM、MEAN、MIN、MAX 共四种更新模式。([#37205](https://github.com/PaddlePaddle/Paddle/pull/37205))
- 新增`paddle.incubate.operators.ResNetUnit` API，用于 ResNet 网络里的卷积、批归一化、shortcut/bottleneck操作融合。([#37109](https://github.com/PaddlePaddle/Paddle/pull/37109))
 

### （2）功能优化

#### API

- `paddle.incubate.FusedTransformerEncoderLayer`，添加 `src_mask=None` 的支持，添加pure fp16的支持。 ([#37229](https://github.com/PaddlePaddle/Paddle/pull/37229))

#### IR(Intermediate Representation)

- 动态图转静态图
	- 使用`@paddle.jit.to_static`装饰单独的 function 时，提供 `train()、eval()` 函数支持切换到 `train、eval` 模式。([#37383](https://github.com/PaddlePaddle/Paddle/pull/37383))


#### 分布式训练
- 异构参数服务器完善任意次切图能力，增加流水线训练功能，提升训练吞吐。([#37446](https://github.com/PaddlePaddle/Paddle/pull/37446))
 

#### 其他

- 针对 `paddle.scatter` 的 ``index`` 越界导致 core dump 的问题，加强了越界检查，并完善对应的报错信息。([#37431](https://github.com/PaddlePaddle/Paddle/pull/37431))


### （3）性能优化

- 优化 `paddle.top_k`，根据 ``k`` 的大小和 ``input_width`` 大小进行选择不同的实现方案，当 k>=75% input_width 时选择 cub 实现，否则选择手写 kernel 实现。([#37325](https://github.com/PaddlePaddle/Paddle/pull/37325))
- 优化`paddle.fluid.optimizer.LarsMomentumOptimizer`，通过 optimizer 算子融合 + [CUDA Cooperative Groups](https://developer.nvidia.com/blog/cooperative-groups/)的方式提高OP性能。([#37109](https://github.com/PaddlePaddle/Paddle/pull/37109))



### （4）问题修复

#### API
- 修复`paddle.nn.ELU` 与 `paddle.nn.functional.elu` 的计算公式，解决 alpha<0 时结果错误的问题；`paddle.nn.functional.elu_`不支持 alpha<0 的场景，在 alpha<0 时会报错。([#37437](https://github.com/PaddlePaddle/Paddle/pull/37437))
- 修复`paddle.slice`反向执行时出现 `out_of_range` 的问题。([#37584](https://github.com/PaddlePaddle/Paddle/pull/37584))
- `paddle.shape` 没有反向，显式设置 ``stop_gradient`` 为 ``True``。([#37412](https://github.com/PaddlePaddle/Paddle/pull/37412))
- `paddle.arange` 没有反向，显式设置 ``stop_gradient`` 为 ``True``。([#37486](https://github.com/PaddlePaddle/Paddle/pull/37486))
- `paddle.shard_index` 在输入数据的最后一维不为1时进行报错提示。([#37421](https://github.com/PaddlePaddle/Paddle/pull/37421))
- 修复 ``paddle.matmul`` 使用int8量化，反量化时维度错误的问题。([#36982](https://github.com/PaddlePaddle/Paddle/pull/36982))
- 修复 `paddle.nn.Dropout` 在 `eval` 模式下不计算梯度的问题。([#37305](https://github.com/PaddlePaddle/Paddle/pull/37305))
- 修复 `paddle.nn.functional.dropout` 在静态图下输入 `Tenor` 形状中有 -1 并指定 drop 该维时报错的问题。([#37223](https://github.com/PaddlePaddle/Paddle/pull/37223))
- 修复RNN类API `paddle.nn.LSTM`,`paddle.nn.GRU`, `paddle.nn.SimpleRNN`在CPU训练时多层RNN（dropout设置为0）反向计算出错的问题。([#37086](https://github.com/PaddlePaddle/Paddle/pull/37086))
- 修复 `paddle.incubate.FusedTransformerEncoderLayer` 反向计算梯度错误、pre_layer_norm 处理不正确、参数处理不正确，漏传参数、 add_bias 计算错误等问题。 ([#37229](https://github.com/PaddlePaddle/Paddle/pull/37229))
- 修复 `paddle.incubate.fused_multi_head_attention` 不支持 ``bias`` 为`None` 的问题。([#37411](https://github.com/PaddlePaddle/Paddle/pull/37411), [#37566](https://github.com/PaddlePaddle/Paddle/pull/37566))
- 修复`paddle.vision.datasets.Cifar10`, `paddle.vision.datasets.Cifar100`加载数据没有顺序的问题。 ([#37528](https://github.com/PaddlePaddle/Paddle/pull/37528))
- 修复一维`Tensor`在使用省略号(...)索引时维度检测异常报错的问题。([#37192](https://github.com/PaddlePaddle/Paddle/pull/37192))
- 修复`Tensor`索引赋值(`setitem`)梯度属性无法传播的问题，详见[issue](https://github.com/PaddlePaddle/Paddle/issues/36902)。([#37028](https://github.com/PaddlePaddle/Paddle/pull/37028))


#### IR(Intermediate Representation)

- 动态图转静态图
	- 动转静后的模型调用 `paddle.flops` 能够正确统计模型参数。([#36852](https://github.com/PaddlePaddle/Paddle/pull/36852))
	- 动转静模块能够正确转换`for i in [1, 2, 3]`循环语句。([#37259](https://github.com/PaddlePaddle/Paddle/pull/37259))

#### 分布式训练

  - `fleet.load_model`: 修复参数服务器模式下模型加载API不可用问题。([#37461](https://github.com/PaddlePaddle/Paddle/pull/37461))
  -  `fleet.save_inference_model`: 修复参数服务器模式下模型保存 dense 参数前，未从 server 端拉取参数的问题。([#37461](https://github.com/PaddlePaddle/Paddle/pull/37461))
 

#### 其他

- 修复动态图 inplace 操作的问题：对一个非叶子节点进行 inplace 操作后，立即执行 backward，该节点及更前的节点的梯度计算错误。([#37420](https://github.com/PaddlePaddle/Paddle/pull/37420))


## 4. 部署方向（Paddle Inference）

### （1）问题修复

- 在明确关闭日志的情况下，进一步去除冗余的调试日志。([#37212](https://github.com/PaddlePaddle/Paddle/pull/37212))
- 修复内存/显存优化策略，避免因不当的内存/显存优化导致预测结果有误或崩溃。([#37324](https://github.com/PaddlePaddle/Paddle/pull/37324), [#37123](https://github.com/PaddlePaddle/Paddle/pull/37123))
- 修复 Transformer 模型的 MultiHead 结构中融合后 QkvToContextPluginDynamicscale 的 scale 计算错误问题，这是由于 cuda 函数的 block 和 thread 设置错误引起的。([#37096](https://github.com/PaddlePaddle/Paddle/pull/37096))
- 将所有的推理OP在in8量化的功能中注册：解决因历史原因有些推理OP没有在int8量化中注册的问题。([#37266](https://github.com/PaddlePaddle/Paddle/pull/37266))


# 2.2.0 Release Note

## 1. 重要更新

我们很高兴的发布飞桨框架2.2.0版本，本版本包含如下重要更新。

### API

- 新增100+个API，包含24个傅里叶变换API、17个线性代数计算 API 等，更好地支持科学计算类、信号处理类模型。
- 新增多种索引类型的支持，新增的索引类型包括：省略号（…）、维度扩增（None）、布尔类型数组（Bool Mask）、整数数组(（list)，以及张量（Tensor) ），可以更加方便的对张量（Tensor）进行操作。
- 新增 `paddle.einsum` API，可以以更加简洁的方式来表达多维张量（Tensor）的计算。
- 动态图混合精度功能增强，新增整个任务使用半精度（float16）训练的方式，主要任务下的计算效率提升20%左右。												

### IR(Intermediate Representation)

- 动态图转静态图：进一步扩充了动静转换支持的语法和场景，现在使用混合精度训练的动态图模型也可以通过 `to_static` 接口一键转换为静态图进行训练或推理部署；另外，对转换后训练的性能进行了优化，通过引入缓存和开启 Pass 等策略，转换后的训练性能相对动态图方式有明显提升。
- Pass 开发：新增 Python 端对静态图IR的改写接口，针对 OP fusion 等子图替换场景可以在 python 中快速完成开发。
- 对算子 Kernel 实现中的底层代码进行了抽象与功能封装，提供高性能的 Block 级 IO 运算和 Compute 运算（Kernel Primitive API）。使用 Kernel Primitive API 进行 Kernel 开发可以更加专注计算逻辑的实现，在保证性能的同时大幅减少代码量，同时实现了算子计算与硬件解耦。

### 分布式

- 混合并行：在静态图已有  4D  混合并行的基础上，进行了流水线执行器等性能优化，千亿模型下训练算力利用达到GPU理论性能峰值的51%；动态图支持了 4D 混合并行能力，千亿模型下功能和性能与静态图持平；增加了自动补全、自动切分等基础功能，具备了基于用户标记的半自动并行能力。
- GPU 参数服务器：千亿模型下，优化数据读取、GPU-PS 构建、SSD 性能，完善流水线等功能，使得整体性能提升一倍，内存占用减少一倍，一台 GPU 机器可替代百台 CPU 机器训练千亿模型。

### 推理部署
- 推理加速：支持最新的 TensorRT 8.x，适配 Nvidia 的硬件新特性进行加速。
- 推理易用性：增加 TensorRT 子图中的动态 Shape 配置的自动推导功能，可选从数据推导出 Shape 的范围，无需琐碎的手动配置，简化了动态 Shape 的使用。


## 2. 不兼容升级

- 针对 `grad` 在路径(`paddle.autograd,grad`, `paddle.grad`) 公开暴露的问题，推荐使用 `paddle.grad`，移除了 `from paddle.autograd import *` ，然后直接调用 `grad` 的方式。([#35579](https://github.com/PaddlePaddle/Paddle/pull/35579))

<table>
<tr>
<th>
2.1
</th>
<th>
2.2
</th>
</tr>

<tr>
<td>
<pre>

```python
>>> import paddle
>>> from paddle.autograd import *
>>> x = paddle.ones(shape=[1], dtype='float32')
>>> x.stop_gradient = False
>>> y = x*x
>>> grad(outputs=[y], inputs=[x])
[Tensor(shape=[1], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
        [2.])]
```
</pre>
</td>

<td>
<pre>

```python
>>> import paddle
>>> from paddle.autograd import *
>>> x = paddle.ones(shape=[1], dtype='float32')
>>> x.stop_gradient = False
>>> y = x*x
>>> grad(outputs=[y], inputs=[x])
NameError: name 'grad' is not defined
>>> paddle.grad(outputs=[y], inputs=[x]) # 改用paddle.grad API
[Tensor(shape=[1], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
       [2.])]
```
</pre>
</td>
</tr>
</table>

- ``Tensor.__setitem__`` 不再支持非 ``int`` 类型的 slice 索引( ``x[start:stop:step] = value`` )。由于 ``float``类型在作为索引时不具有数学意义（ 如 ``start`` 为 0.5 时如何确定具体索引的位置）且容易导致一些未知行为，所以本次更新中我们把 slice 索引的数据类型限定为 ``int``，使用 ``float`` 的 slice 索引将报错。([#35701](https://github.com/PaddlePaddle/Paddle/pull/35701))



<table>
<tr>
<th>
2.1
</th>
<th>
2.2
</th>
</tr>

<tr>
<td>
<pre>

```python
>>> import paddle
>>> x = paddle.to_tensor([1, 2, 3, 4])
>>> start = paddle.zeros([1])
>>> stop = paddle.zeros([1]) + 2
>>> step = paddle.ones([1])
>>> x[start:stop:step] = 0 # start,stop,step 支持float类型Tensor
>>> x 
```
</pre>
</td>

<td>
<pre>

```python
>>> import paddle
>>> x = paddle.to_tensor([1, 2, 3, 4])
>>> start = paddle.zeros([1])
>>> stop = paddle.zeros([1]) + 2
>>> step = paddle.ones([1])
>>> x[start:stop:step] = 0
ValueError: (InvalidArgument) Currently, the type of tensor in slice indices only allows int32 and int64, please check the type of index tensor.

>>> # 须改为如下代码：
>>> start = paddle.zeros([1], dtype='int32')
>>> stop = paddle.zeros([1], dtype='int32') + 2
>>> step = paddle.ones([1], dtype='int32')
>>> x[start:stop:step] = 0 # start,stop,step 必须为integer类型Tensor
>>> x
Tensor(shape=[4], dtype=int64, place=CUDAPlace(0), stop_gradient=True,
       [0, 0, 3, 4])
```
</pre>
</td>
</tr>
</table>


- 为动态图``Tensor.__setitem__`` 中加入 inplace 调用合法性检测，不满足检测的赋值代码会报错（检测逻辑：当 ``Tensor`` 为叶节点并且`stop_gradient`为 ``False`` 时，``Tensor`` 赋值操作将被拦截并报错）。由于 ``tensor[index]=value``的执行会覆盖 ``Tensor`` 中原来的值，是 ``Tensor`` 的 inplace 操作，如果 ``Tensor`` 是计算图中的一个叶节点并且需要计算梯度时，进行 ``Tensor`` 的赋值操作会使该 ``Tensor`` 反向梯度的计算出现问题，属于非法的 inplace 操作。所以本次更新加入了对这种操作的检测与拦截，当前使用 ``tensor[index]=value`` 方式赋值的代码都会检测是否满足 inplace 操作的要求，不满足将会报错。  ([#35701](https://github.com/PaddlePaddle/Paddle/pull/35701))
	- 示例：使用` weight[index]=value `方式的参数初始化代码调整，`self.weight`属于叶节点且需要计算梯度，不能使用inplace操作（会影响反向梯度值计算），但初始化赋值本身不需要反向计算过程，所以在明确不需要反向计算时，可以使用`no_grad`关闭梯度计算后再进行赋值。


<table>
<tr>
<th>
2.1
</th>
<th>
2.2
</th>
</tr>

<tr>
<td>
<pre>

```python
>>> import paddle
>>> class MyLayer(paddle.nn.Layer):
...     def __init__(self):
...         super(MyLayer, self).__init__()
...         self.weight = self.create_parameter(...)
...         self.weight[index] = 1.0
...
```
</pre>
</td>

<td>
<pre>

```python
>>> import paddle
class MyLayer(paddle.nn.Layer):
...     def __init__(self):
...         super(MyLayer, self).__init__()
...         self.weight = self.create_parameter(...)
...         with paddle.no_grad(): # 关闭梯度计算后可进行赋值
...             self.weight[index] = 1.0
```
</pre>
</td>
</tr>
</table>

- 针对`paddle.sum` 输入类型为 ``bool`` 时，输出类型也为``bool``，行为与``numpy.sum`` 不一致问题，进行了不兼容升级，升级后输出类型为``int64``，与 ``numpy.sum`` 保持一致。([#34313](https://github.com/PaddlePaddle/Paddle/pull/34313))


<table>
<tr>
<th>
2.1
</th>
<th>
2.2
</th>
</tr>

<tr>
<td>
<pre>

```python
>>> import paddle
>>> import numpy as np
>>> np_arr = np.ones((2, 3), dtype='bool')
>>> pd_arr = paddle.to_tensor(np_arr)
>>> pd_sum = pd_arr.sum(0)
>>> pd_sum.dtype
paddle.bool
```
</pre>
</td>

<td>
<pre>

```python
>>> import paddle
>>> import numpy as np
>>> np_arr = np.ones((2, 3), dtype='bool')
>>> pd_arr = paddle.to_tensor(np_arr)
>>> pd_sum = pd_arr.sum(0)
>>> pd_sum.dtype
paddle.int64
```
</pre>
</td>
</tr>
</table>


- 针对``paddle.to_tensor``在输入 ``data`` 为 ``Tensor`` 时不拷贝 ``Tensor`` 导致 ``stop_gradient`` 属性可能被错误修改的问题，优化了该情况下的 ``Tensor`` 拷贝行为。原实现中，当 ``data`` 为 ``Tensor`` 且 ``dtype`` 和 ``place`` 不改变时，会直接返回 ``data``（即不发生拷贝）并修改 ``data.stop_gradient`` 属性。该行为会导致原来的计算图 ``data`` 的反向传播出现问题。新实现中，上述情况下，``paddle.to_tensor`` 会拷贝一个新的 ``Tensor`` 且返回，不会修改原 ``data`` 的 ``stop_gradient`` 属性。([#33335](https://github.com/PaddlePaddle/Paddle/pull/33335)) 

<table>
<tr>
<th>
2.1
</th>
<th>
2.2
</th>
</tr>

<tr>
<td>
<pre>

```python
>>> import paddle
>>> x = paddle.rand([2,3])
>>> x.stop_gradient = False
>>> y = paddle.to_tensor(x)
>>> print(id(x) == id(y)) # True
>>> print(x.stop_gradient, y.stop_gradient) # True True
```
</pre>
</td>

<td>
<pre>

```python
>>> import paddle
>>> x = paddle.rand([2,3])
>>> x.stop_gradient = False
>>> y = paddle.to_tensor(x)
>>> print(id(x) == id(y)) # False
>>> print(x.stop_gradient, y.stop_gradient) # False True
```
</pre>
</td>
</tr>
</table>


## 3. 训练框架（含分布式）

### （1）新功能
#### API
- 新增线性代数计算API``paddle.linalg.*``
 - 新增 ``paddle.linalg.svd``，支持对多维 ``Tensor`` 进行奇异值分解。([#34953](https://github.com/PaddlePaddle/Paddle/pull/34953)) 
	- 新增 ``paddle.linalg.cond``，支持根据范数种类 ``p`` 计算一个或一批矩阵的条件数。([#35140](https://github.com/PaddlePaddle/Paddle/pull/35140)) 
	- 新增 ``paddle.linalg.matrix_rank``，支持计算多维矩阵 ``Tensor``的秩。 ([#34823](https://github.com/PaddlePaddle/Paddle/pull/34823)) 
	- 新增 ``paddle.linalg.eigvals``，支持计算一般方阵的特征值。 ([#35720](https://github.com/PaddlePaddle/Paddle/pull/35720), [#35909](https://github.com/PaddlePaddle/Paddle/pull/35720))
	- 新增 ``paddle.linalg.eigh``，支持计算复数厄米特矩阵或者实数对称矩阵的特征值和特征向量。([#34990](https://github.com/PaddlePaddle/Paddle/pull/34990), [#35916](https://github.com/PaddlePaddle/Paddle/pull/35916), [#35812](https://github.com/PaddlePaddle/Paddle/pull/35812), [#36091](https://github.com/PaddlePaddle/Paddle/pull/36091),[#35919](https://github.com/PaddlePaddle/Paddle/pull/35919)) 
	- 新增 ``paddle.linalg.det``， 支持计算多维矩阵的行列式值。([#34992](https://github.com/PaddlePaddle/Paddle/pull/34992)) 
	- 新增 ``paddle.linalg.slogdet``，支持计算多维矩阵行列式值的符号值与自然对数值。([#34992](https://github.com/PaddlePaddle/Paddle/pull/34992))
	- 新增 ``paddle.linalg.pinv``，支持计算多维矩阵 ``Tensor`` 的伪逆矩阵。([#35804](https://github.com/PaddlePaddle/Paddle/pull/35804))
	- 新增 ``paddle.linalg.multi_dot``，支持多个矩阵连乘的计算。([#35224](https://github.com/PaddlePaddle/Paddle/pull/35224))
	- 新增 ``paddle.linalg.solve``，支持计算线性方程组的解。([#35715](https://github.com/PaddlePaddle/Paddle/pull/35715))
	- 新增``paddle.linalg.matrix_power``，支持矩阵的幂运算操作。([#34667](https://github.com/PaddlePaddle/Paddle/pull/34667))
	-  新增`paddle.linalg.eigvalsh`，用于计算厄米特矩阵或者实数对称矩阵的特征值。([#36680](https://github.com/PaddlePaddle/Paddle/pull/36680))
	- 新增`paddle.linalg.eig`，用于计算一般方阵的特征值和特征向量。([#35674](https://github.com/PaddlePaddle/Paddle/pull/35674))
	- 新增`paddle.linalg.qr`，用于计算矩阵的QR分解（暂不支持反向）。([#36627](https://github.com/PaddlePaddle/Paddle/pull/36627))
- 新增傅里叶变换相关API ([#35665](https://github.com/PaddlePaddle/Paddle/pull/35665)) 
    - 新增快速傅立叶变换系列函数
        - 可微分的 1d 到 nd 复数到复数快速傅里叶变换。(``paddle.fft.fft``, ``paddle.fft.fft2``, ``paddle.fft.fftn``, ``paddle.fft.ifft``, ``paddle.fft.ifft2``, ``paddle.fft.ifftn``)
        - 可微分的 1d 到 nd 实数到复数快速傅里叶变换。(``paddle.fft.rfft``, ``paddle.fft.rfft2``, ``paddle.fft.rfftn``, ``paddle.fft.ihfft``, ``paddle.fft.ihfft2``, ``paddle.fft.ihfftn``)
        - 可微分的 1d 到 nd 复数到实数快速傅里叶变换。 (``paddle.fft.hfft``, ``paddle.fft.hfft2``, ``paddle.fft.hfftn``, ``paddle.fft.irfft``, ``paddle.fft.irfft2``, ``paddle.fft.irfftn``)
        - fft 相关的辅助函数。(``paddle.fft.fftfreq``, ``paddle.fft.rfftfreq``, ``paddle.fft.fftshift``, ``paddle.fft.ifftshift``)

    - 新增短时傅里叶变换相关函数
        - 短时傅里叶变换。(``paddle.signal.stft``)
        - 短时傅里叶逆变换。(``paddle.signal.istft``)

- 新增高层API
	- 新增 ``paddle.vision.ops.roi_pool`` 和 ``paddle.vision.ops.RoIPool``，支持检测任务中 RoI 区域池化操作。 ([#36154](https://github.com/PaddlePaddle/Paddle/pull/36154))
	-  新增 ``paddle.vision.ops.roi_align`` 和 ``paddle.vision.ops.RoIAlign``，支持检测任务中 RoI 区域 Align 操作。([#36207](https://github.com/PaddlePaddle/Paddle/pull/36207))
	- 新增 ``paddle.vision.ops.psroi_pool`` 和 ``paddle.vision.ops.PSRoIPool``，支持检测任务中位置敏感的 RoI 区域池化操作。 ([#36111](https://github.com/PaddlePaddle/Paddle/pull/36111))
	- 新增 ``paddle.vision.models.vgg19`` 预训练权重。 ([#35788](https://github.com/PaddlePaddle/Paddle/pull/35788))
	- 新增 ``paddle.vision.datasets.*`` 中数据集 API 下载进度条。([#33302](https://github.com/PaddlePaddle/Paddle/pull/33302))
	- 新增 ``paddle.Model.predict`` 参数 ``verbose``，支持是否显示日志。([#33405](https://github.com/PaddlePaddle/Paddle/pull/33405))
	- 新增 ``paddle.hub`` 下载选项 `wget` 方式。([#33379](https://github.com/PaddlePaddle/Paddle/pull/33379))
	-  新增 ``paddle.Model`` 动态图模式下梯度累加功能。([#32702](https://github.com/PaddlePaddle/Paddle/pull/32702))
	- 新增 ``paddle.Model.fit`` 和 ``paddle.Model.evaluate``  动态图模式下 ``num_iters`` 参数，控制训练迭代轮数。([#33986](https://github.com/PaddlePaddle/Paddle/pull/33986))
	- 新增 ``paddle.vision.ops.yolo_box`` 参数 ``iou_aware`` 和 ``iou_aware_factor``，支持 YoloBox 使用预测的 IOU 作为置信度的因子。([#33400](https://github.com/PaddlePaddle/Paddle/pull/33400))
	- 新增 ``paddle.summary`` 参数``input``，支持给定输入。([#34165](https://github.com/PaddlePaddle/Paddle/pull/34165))
	- 新增`paddle.text.viterbi_decode`，支持动态图下CPU、GPU的Viterbi解码功能。([#35778](https://github.com/PaddlePaddle/Paddle/pull/35778))

- 新增组网类 API
	- 新增`paddle.nn.functional.sparse_attention`，用于计算稀疏的Transformer Attention模块。([#35757](https://github.com/PaddlePaddle/Paddle/pull/35757))
	- 新增 ``paddle.nn.MaxUnPool2D`` 和 ``paddle.nn.functional.max_unpool2d``，支持根据输入的input和最大值位置计算出池化的逆结果。([#35056](https://github.com/PaddlePaddle/Paddle/pull/35056))
	- 新增 ``paddle.nn.functional.gumbel_softmax``，支持 ``gumbel softmax`` 采样。([#35506](https://github.com/PaddlePaddle/Paddle/pull/35506), [#36065](https://github.com/PaddlePaddle/Paddle/pull/36065), [#36094](https://github.com/PaddlePaddle/Paddle/pull/36094))
	- 新增 ``paddle.nn.functional.class_center_sample``，支持 PartialFC 类中心采样功能。([#34106](https://github.com/PaddlePaddle/Paddle/pull/34106))
	- 新增 ``paddle.nn.functional.margin_cross_entropy``，支持 ArcFace，CosFace，SphereFace 等 MarginLoss 功能。([#34247](https://github.com/PaddlePaddle/Paddle/pull/34247))
	- ``paddle.nn.AvgPool2D``支持二阶导数。([#35388](https://github.com/PaddlePaddle/Paddle/pull/35388))
	- ``paddle.nn.Linear、paddle.matmul、paddle.mm`` 支持二阶导数。[#35428](https://github.com/PaddlePaddle/Paddle/pull/35428)
	- ``paddle.nn.GroupNorm``支持 (N, C, *) 形式的输入。([#34773](https://github.com/PaddlePaddle/Paddle/pull/34773))
	- 新增 ``paddle.nn.BatchNorm1D/2D/3D`` 在 ``x.stop_gradient=True`` 的条件下计算反向。([#34102](https://github.com/PaddlePaddle/Paddle/pull/34102))
	- 新增 ``paddle.nn.Dropout, paddle,nn.Dropout2D/3D`` 在 ``model.eval``模式下计算反向 。([#35122](https://github.com/PaddlePaddle/Paddle/pull/35122))

- 新增硬件相关API
	- 新增`paddle.device.cuda.Stream`,`paddle.device.cuda.Event`,`paddle.device.cuda.current_stream`,`paddle.device.cuda.synchronize` ， 支持在Python端对CUDA的event和 stream进行同步操作。([#32460](https://github.com/PaddlePaddle/Paddle/pull/32460))
	- 新增 ``paddle.device.cuda.device_count``，支持返回当前可用GPU数量。([#34811](https://github.com/PaddlePaddle/Paddle/pull/34811))
	- 新增 ``paddle.device.cuda.empty_cache``，支持清理空闲的显存。([#35427](https://github.com/PaddlePaddle/Paddle/pull/35427))
	- 新增 ``paddle.device.cuda.get_device_properties``，支持返回给定的设备属性。([#35875](https://github.com/PaddlePaddle/Paddle/pull/35875))
	- 新增 ``paddle.device.cuda.stream_guard``，用于动态图下 CUDA Stream的灵活切换。([#35623](https://github.com/PaddlePaddle/Paddle/pull/35623))
	- 新增`paddle.device.cuda.get_device_name`，支持返回给定设备的名称。([#36172](https://github.com/PaddlePaddle/Paddle/pull/36172))
	- 新增`paddle.device.cuda.get_device_capability`，支持返回给定设备计算能力的版本号。([#36172](https://github.com/PaddlePaddle/Paddle/pull/36172))
	- 新增`paddle.framework.core.async_read`和`paddle.framework.core.async_write`，可支持非默认 CUDA `Stream`下`CUDAPinnedPlace` 和 `CUDAPlace` 的 `Tensor` 数据异步读写。([#36501](https://github.com/PaddlePaddle/Paddle/pull/36501))

- 新增Tensor操作API
 - 新增`paddle.tensordot`，支持对高维张量做缩并(Tensor Contraction)运算。([#36454](https://github.com/PaddlePaddle/Paddle/pull/36454))
 - 新增`paddle.bincount`，支持对一维张量内元素进行计数。([#36709](https://github.com/PaddlePaddle/Paddle/pull/36709))
 - 新增 `paddle.broadcast_tensors` ，支持对一组 `Tensor` 进行广播操作。([#33294](https://github.com/PaddlePaddle/Paddle/pull/33294), [#34874](https://github.com/PaddlePaddle/Paddle/pull/34874))
 - 新增 `paddle.einsum` 。([#33821](https://github.com/PaddlePaddle/Paddle/pull/33821))
 - 增强``paddle.tensor.gradient``接口，支持sigmoid_op的二阶求导算子。([#32971](https://github.com/PaddlePaddle/Paddle/pull/32971))
 - 新增 ``paddle.searchsorted``，支持在有序``Tensor``中查找给定值的索引。([#35159](https://github.com/PaddlePaddle/Paddle/pull/35159))
 - 新增 ``paddle.unique_consecutive`` ，支持将 ``Tensor`` 中连续重复的元素进行去重，返回连续不重复的``Tensor``。([#34334](https://github.com/PaddlePaddle/Paddle/pull/34334))
 - 新增  ``paddle.diagflat``，支持返回以输入 ``Tensor`` 的元素为对角线的对角矩阵。([#33334](https://github.com/PaddlePaddle/Paddle/pull/33334))
 - 新增 ``paddle.lgamma``，支持逐元素计算 ``Tensor`` 的 ``lgamma`` 函数值。([#33913](https://github.com/PaddlePaddle/Paddle/pull/32913))
 - 新增 ``paddle.digamma``，支持逐元素计算 ``Tensor`` 的 ``digamma`` 函数值。([#33278](https://github.com/PaddlePaddle/Paddle/pull/33278))
 - 新增 ``paddle.neg``，支持逐元素计算 ``Tensor`` 的相反数值。([#33248](https://github.com/PaddlePaddle/Paddle/pull/33248))
 - 新增 ``paddle.cumprod``，支持根据给定维度计算 ``Tensor`` 累乘。([#35185](https://github.com/PaddlePaddle/Paddle/pull/35185))
 - 新增 ``paddle.atan2`` ，支持逐元素的 ``arctangent`` 运算，通过符号确定象限。([#33067](https://github.com/PaddlePaddle/Paddle/pull/33067))
 - 新增 ``paddle.expm1``，支持逐元素进行以 ``exp(x)-1`` 运算。 ([#33066](https://github.com/PaddlePaddle/Paddle/pull/33066))
 - 新增 ``paddle.trunc``，支持对输入的 ``Tensor`` 进行截断整数值。([#33371](https://github.com/PaddlePaddle/Paddle/pull/33371))
 - 新增 ``paddle.diagonal``，支持提取输入的 ``Tensor`` 的对角线元素。 ([#33586](https://github.com/PaddlePaddle/Paddle/pull/33586)) 
 - 新增``paddle.utils.dlpack``，包含： ``paddle.utils.dlpack.to_dlpack`` 和 ``paddle.utils.dlpack.from_dlpack``，利用 ``DLPack`` 支持不同框架间的 ``Tensor`` 传输。([#35067](https://github.com/PaddlePaddle/Paddle/pull/35067))
 - 新增 ``paddle.Tensor.uniform_``, 支持使用服从均匀分布的随机数原地填充一个``Tensor``。([#33394](https://github.com/PaddlePaddle/Paddle/pull/33934))
 - 新增 ``paddle.Tensor.T``，对 N-D Tensor 会进行转置，返回一个与原 Tensor 的shape相反的Tensor。([#35379](https://github.com/PaddlePaddle/Paddle/pull/35379)) 
 - 新增 ``paddle.Tensor`` 魔法操作符：&（按位与）、| （按位或）、^ （按位异或）、~（按位取反）。 ([#33524](https://github.com/PaddlePaddle/Paddle/pull/33524))
 - 新增 `paddle.Tensor.fill_`、`paddle.Tensor.zero_`，原地修改Tensor中的值，分别使用固定值填充、使用全零填充。([#33829](https://github.com/PaddlePaddle/Paddle/pull/33829)) 
 - 新增 `paddle.Tensor.fill_diagonal`、`paddle.Tensor.fill_diagonal` ,修改Tensor对角线元素值。([#34460](https://github.com/PaddlePaddle/Paddle/pull/34460)) 
 - 新增 `paddle.Tensor.fill_diagonal_tensor_`，对Tensor两个指定坐标轴的对角线与其他坐标轴形成的子Tensor进行整体修改。([#34515](https://github.com/PaddlePaddle/Paddle/pull/34515)) 
 - 动静态图 ``Tensor`` 新增多种索引类型的支持，包括：省略号（...）、维度扩增（None）、布尔类型数组（Bool Mask）、整数数组（list）以及张量（Tensor）。
    - 省略号（...）索引：`X[..., 0]` 。([#34267](https://github.com/PaddlePaddle/Paddle/pull/34267), [#32876](https://github.com/PaddlePaddle/Paddle/pull/32876))
    - 维度扩增（None）索引： `X[None, :]` 。([#34338](https://github.com/PaddlePaddle/Paddle/pull/34338), [#34442](https://github.com/PaddlePaddle/Paddle/pull/34442),  [#34877](https://github.com/PaddlePaddle/Paddle/pull/34877),  [#34911](https://github.com/PaddlePaddle/Paddle/pull/34911),  [#33001](https://github.com/PaddlePaddle/Paddle/pull/33001))
	 - 布尔类型数组（Bool Mask）索引：`X[X > 0] = 0` 。 ([#35026](https://github.com/PaddlePaddle/Paddle/pull/35026),  [#35133](https://github.com/PaddlePaddle/Paddle/pull/35133),  [#33298](https://github.com/PaddlePaddle/Paddle/pull/33298))
	 - 整数数组（list）索引：`X[[1, 0], [0]]` 。([#34824](https://github.com/PaddlePaddle/Paddle/pull/34824), [#33000](https://github.com/PaddlePaddle/Paddle/pull/33000),  [#35404](https://github.com/PaddlePaddle/Paddle/pull/35404))
	 - 张量（Tensor）索引：`X[panddle.to_tensor([0, 1], [1, 0])]` 。([#34824](https://github.com/PaddlePaddle/Paddle/pull/34824))

- 新增分布式相关API
    - 新增 ``paddle.distributed.utils.global_scatter`` 和 `paddle.distributed.utils.global_gather`，支持 MOE 有条件分发数据，`global_scatter`会根据条件将数据分发到所有卡上，然后`global_gather`则会将数据根据条件从所有 GPU 卡上收集数据。([#35546](https://github.com/PaddlePaddle/Paddle/pull/35546))

- 新增其他的API
    -  新增 ``paddle.disable_signal_handler`` ，支持关闭PaddlePaddle中信号捕捉机制，从而使得用户可以同时使用Paddle与TVM。([#34577](https://github.com/PaddlePaddle/Paddle/pull/34577))
    - 新增  ``paddle.incubate.softmax_mask_fuse ``，支持加速 Transformer 架构的 softmax 与 mask 的运算速度。([#33841](https://github.com/PaddlePaddle/Paddle/pull/33841))
    - 新增  ``paddle.incubate.softmax_mask_fuse_upper_triangle ``，支持加速 GPT 版本的 Transformer 架构的 softmax 与 mask 的运算速度。([#33981](https://github.com/PaddlePaddle/Paddle/pull/33981))
    - 新增  ``paddle.static.ExponentialMovingAverage``，支持用指数衰减计算参数的滑动平均值。([#35673](https://github.com/PaddlePaddle/Paddle/pull/35673))
    - 新增 `` paddle::Tensor::slice`` C++ API， 支持 slice 操作，允许用户对外部 Tensor 切片操作。([#34227](https://github.com/PaddlePaddle/Paddle/pull/34227))
    - 新增``paddle.incubate.segment_*``系列API，包含 ``paddle.incubate.segment_sum, paddle.incubate.segment_mean,  paddle.incubate.segment_max, paddle.incubate.segment_min``。支持对`Tensor`按照分段求和、求均值、求最大值、求最小值。 ([#35759](https://github.com/PaddlePaddle/Paddle/pull/35759))
    - 新增`paddle.version.cuda`和`paddle.version.cudnn`，用于获取 paddle 安装包所使用的 `CUDA`和 `cuDNN`的版本号。([#36556](https://github.com/PaddlePaddle/Paddle/pull/36556))

#### IR(Intermediate Representation)
- 动态图转静态图 
    - 新增动转静转写报错类型识别，并给出修改建议。 ([#35648](https://github.com/PaddlePaddle/Paddle/pull/35648)) 
    - 新增对混合精度训练功能支持，``@to_static`` c支持一键转为静态图混合精度训练模式。 ([#34562](https://github.com/PaddlePaddle/Paddle/pull/34562))
	- ``@to_static`` 中新增 ``build_strategy`` 参数，支持动转静后自定义开启相关 `Pass` 优化策略加速模型训练，如算子融合等。 ([#34347](https://github.com/PaddlePaddle/Paddle/pull/34347))
	- 增加`a, b = static_variable` 的支持。([#33499](https://github.com/PaddlePaddle/Paddle/pull/33499))
	- 新增二阶导能力支持。([#33110](https://github.com/PaddlePaddle/Paddle/pull/33110))

- Program和Graph互转 ：``Program`` 和 ``Graph``是 飞桨框架底层用来表达计算的中间表示，对于飞桨的开发者而言，有时需要将 ``Program`` 和 ``Graph``互相转化来进行计算处理。本功能添加了 ``Program`` 和 ``Graph`` 互转相关能力。
    - 开发完善 ``Program`` 和 ``Graph`` 相互转换功能。 ([#33949](https://github.com/PaddlePaddle/Paddle/pull/33949))
    - 为了支持 `while` 等控制流节点，飞桨框架的 `Program` 中除了主 `block` 外，还可能包含多个子 `block`。之前 `Program` 转 `Graph` 的过程中，只将主 `block` 转化为 `Graph`，这里改进 `Graph`，支持表达子 `block`，实现完整的 `Program` 转 `Graph`。([#33320](https://github.com/PaddlePaddle/Paddle/pull/33320))
    - 提供分析 `Program` 中控制流需要的依赖辅助函数。 ([#33439](https://github.com/PaddlePaddle/Paddle/pull/33439))
    - `Program` 和 `Graph` 相互转换后保留训练所需要的 `stop_gradient` ,  `persistable` 属性值。([#33771](https://github.com/PaddlePaddle/Paddle/pull/33771)) 
    - 原 `Pass` 只处理主`Graph`，忽略子图，现`Pass` 支持处理主 `Graph`及其所有子图。 ([#34158](https://github.com/PaddlePaddle/Paddle/pull/34158)) 
    - 处理了在预测情况下 `Program` 和 `Graph` 互转的一些拓扑排序问题。([#34121](https://github.com/PaddlePaddle/Paddle/pull/34121), [#34521](https://github.com/PaddlePaddle/Paddle/pull/34521))

- Pass开发
    - 新增 Python 侧针对 fusion 等子图替换场景下的 Pass 开发方式。([#35708](https://github.com/PaddlePaddle/Paddle/pull/35708), [#35602](https://github.com/PaddlePaddle/Paddle/pull/35602))

- Kernel Primitive API	
    - 对算子 Kernel 实现中的底层代码进行了抽象与功能封装，提供高性能的 Block 级 IO 运算和 Compute 运算。使用 Kernel Primitive API 进行 Kernel 开发可以更加专注计算逻辑的实现，在保证性能的同时大幅减少代码量，同时实现了算子计算与硬件解耦。([#34672](https://github.com/PaddlePaddle/Paddle/pull/34672),  [#35075](https://github.com/PaddlePaddle/Paddle/pull/35075),  [#34456](https://github.com/PaddlePaddle/Paddle/pull/34456),  [#35282](https://github.com/PaddlePaddle/Paddle/pull/35282),  [#35743](https://github.com/PaddlePaddle/Paddle/pull/35743),  [#34208](https://github.com/PaddlePaddle/Paddle/pull/34208))
    - 在 Kernel Primitive API中添加一元和二元计算Functor共13个。 ([#36418](https://github.com/PaddlePaddle/Paddle/pull/36418))
    - 修改 Kernel Primitive API 中 ReadData 实现方式，修复`NX !=1`访存越界的问题。 ([#36373](https://github.com/PaddlePaddle/Paddle/pull/36373))

#### 混合精度训练
- 动态图混合精度功能增强，新增整个任务使用半精度（float16）训练的方式，主要任务下的计算效率提升20%左右。 ([#35521](https://github.com/PaddlePaddle/Paddle/pull/35521))
- 动态图混合精度 ``paddle.amp.GradScaler`` 新增 ``get`` 和 ``set`` 方法，方便用户设置。([#33835](https://github.com/PaddlePaddle/Paddle/pull/33835))
- 动态图混合精度 ``paddle.amp.GradScaler`` 新增 ``state_dict`` 和 ``load_state_dict`` 方法。 ([#34300](https://github.com/PaddlePaddle/Paddle/pull/34300))
- 动态图混合精度拆分 ``minimize``为 ``step`` + ``update`` ；并新增 ``unscale``方法。 ([#35927](https://github.com/PaddlePaddle/Paddle/pull/35927))
- 动态图混合精度训练支持 param group。([#34899](https://github.com/PaddlePaddle/Paddle/pull/34899))
- 静态图混合精度训练支持梯度裁剪。 ([#33565](https://github.com/PaddlePaddle/Paddle/pull/33565))


#### 分布式训练
- 分布式训练基础功能
    - 新增 `paddle.DataParallel.no_sync`，实现动态图数据并行下暂停多卡通信和梯度同步。([#34740](https://github.com/PaddlePaddle/Paddle/pull/34740)) 
    - 新增 `paddle.distributed.launch` 启动方式对容错的支持，实现 `collective` 模式下的节点容错功能。 ([#33369](https://github.com/PaddlePaddle/Paddle/pull/33369),  [#34572](https://github.com/PaddlePaddle/Paddle/pull/34572))
	- 分布式训练API `paddle.static.Executor.train_from_dataset, paddle.static.Executor.infer_from_dataset` 新增dump功能训练过程中模型的参数和中间变量的功能。[#34457](https://github.com/PaddlePaddle/Paddle/pull/34457) 
	- 混合并行支持模型并行与数据并行的组合。([#34377](https://github.com/PaddlePaddle/Paddle/pull/34377))
	- 新增分布式策略`gradient scale`选项，用户可以指定`gradient scale`的方式：`avg`、`sum`或者自定义。([#33862](https://github.com/PaddlePaddle/Paddle/pull/33862))
	- 新增 `paddle.distributed.parallel_with_gloo`，支持 CPU barrier 操作。([#34671](https://github.com/PaddlePaddle/Paddle/pull/34671))
	- GPU 参数服务器新增训练 profiler 功能。([#32640](https://github.com/PaddlePaddle/Paddle/pull/32640))
	- GPU 参数服务器新增流水线功能，训练性能提升可40%。[#33159](https://github.com/PaddlePaddle/Paddle/pull/33159)  
	- 静态图混合并行添加 `dp_as_optimizer_sharding` 实验性功能，可将数据并行作为优化器参数分片并行，节约优化器状态显存占用。([#35593](https://github.com/PaddlePaddle/Paddle/pull/35593))
	- 静态图流水线并行执行器支持 `LRScheduler`。([#34402](https://github.com/PaddlePaddle/Paddle/pull/34402))
	- 新增`paddle.fluid.core.GraphPyClient.set_node_feat`,支持用户在图引擎客户端设置图节点特征,支持多种类型特征存储。([#34994](https://github.com/PaddlePaddle/Paddle/pull/34994))
	- 提升图引擎图节点邻居采样算法的性能，优化图游走算法的执行。([#34088](https://github.com/PaddlePaddle/Paddle/pull/34088))
	- 模型并行接口`paddle.distributed.fleet.meta_parallel.ColumnParallelLinear`、`paddle.distributed.fleet.meta_parallel.RowParallelLinear`、`paddle.distributed.fleet.meta_parallel.VocabParallelEmbedding`、`paddle.distributed.fleet.meta_parallel.ParallelCrossEntropy`实现动静统一。([#33700](https://github.com/PaddlePaddle/Paddle/pull/33700),  [#33411](https://github.com/PaddlePaddle/Paddle/pull/33411))
	- 新增分布式模型并行cpu `c_embedding` op。([#35467](https://github.com/PaddlePaddle/Paddle/pull/35467))
	- 已修改为新增分布式通信初始化阶段gen_comm_id时得到 gethostbyname 的重试机制。([#34855](https://github.com/PaddlePaddle/Paddle/pull/34855))
	- 新增 `fleet` 梯度更新时的开关配置 `scale_sparse_gradient_with_batch_size`，决定梯度是否乘以 `batch_size`。  ([#34893](https://github.com/PaddlePaddle/Paddle/pull/34893))

- 动态图混合并行
    - 在动态图分布式数据并行场景下，新增 `paddle.distributed.fleet.dygraph_optimizer.DygraphShardingOptimizer` 接口，通过在不同卡间切分优化器状态优化显存占用，支持更大的模型或batch size。 ([#33633](https://github.com/PaddlePaddle/Paddle/pull/33633))
    - 动态图 Sharding 支持 MP-PP-DP， 实现动态图 4D 混合并行。([#35580](https://github.com/PaddlePaddle/Paddle/pull/35580))
    - 动态图 Recompute 支持混合精度计算。([#33251](https://github.com/PaddlePaddle/Paddle/pull/33251))
    - 流水线并行支持 1f1b 调度策略，用于节约运行期显存。([#34483](https://github.com/PaddlePaddle/Paddle/pull/34483))
    - 动态图3D混合并行支持 recompute 策略，支持offload功能。 ([#34607](https://github.com/PaddlePaddle/Paddle/pull/34607) [#35588](https://github.com/PaddlePaddle/Paddle/pull/35588))
    - 动态图3D混合并行支持模型保存和加载。 ([#34768](https://github.com/PaddlePaddle/Paddle/pull/34768))
    - 针对模型并行+流水线并行场景，新增scatter-gather方案，优化跨机通信性能。 ([#34130](https://github.com/PaddlePaddle/Paddle/pull/34130))
    - 流水线并行支持根据 Layer 数量的切分方式，保证切分更加均衡。 ([#34207](https://github.com/PaddlePaddle/Paddle/pull/34207))
    - 流水线并行支持自动混合精度。([#33951](https://github.com/PaddlePaddle/Paddle/pull/33951))
    - 流水线并行添加`paddle.distributed.fleet.meta_parallel.SharedLayerDesc`的组网描述， 用于支持参数共享的组网方式。([#33578](https://github.com/PaddlePaddle/Paddle/pull/33578))
    - 张量并行添加 `paddle.distributed.fleet.meta_parallel.ParallelCrossEntropy`，支持交叉熵Loss的张量并行计算方式。([#33401](https://github.com/PaddlePaddle/Paddle/pull/33401))
    - `paddle.DataParallel`添加`find_unused_parameters`接口，用于数据并行模式下，支持模型中使用控制流的情况。([#32826](https://github.com/PaddlePaddle/Paddle/pull/32826))
    - 数据并行模式添加端口等待功能，解决端口冲突问题。([#34207](https://github.com/PaddlePaddle/Paddle/pull/34207))

- 静态图混合并行
    - 支持流水线并行下 fuse grad merge 的功能，通过 `distributed_strategy.fuse_grad_merge` 开关控制，性能提升约5%。([#35004](https://github.com/PaddlePaddle/Paddle/pull/35004))
    - 支持混合并行开启 dp 下 fuse allreduce sum功能，性能提升约3%。([#34480](https://github.com/PaddlePaddle/Paddle/pull/34480))
	
- 自动并行
    - 新增自动并行 `shard_tensor`，`shard_op` 接口。([#33804](https://github.com/PaddlePaddle/Paddle/pull/33804), [#35765](https://github.com/PaddlePaddle/Paddle/pull/35765))，支持基于用户标记的半自动并行。
    - 新增自动补全分布式属性功能，支持基于用户已标记的分布式属性补全所有未标记的分布式属性。 ([#34813](https://github.com/PaddlePaddle/Paddle/pull/34813))
    - 新增自动切分串行 `Program` 功能。([#35117](https://github.com/PaddlePaddle/Paddle/pull/35117))
    - 实现自动并行对 Fleet API 的适配。([#35483](https://github.com/PaddlePaddle/Paddle/pull/35483))


#### 其他

- 模型量化
    - 新增动态图离线量化功能。([#33445](https://github.com/PaddlePaddle/Paddle/pull/33445),  [#33898](https://github.com/PaddlePaddle/Paddle/pull/33898), [#33962](https://github.com/PaddlePaddle/Paddle/pull/33962),  [#35015](https://github.com/PaddlePaddle/Paddle/pull/35015))
    - 重构动态图量化训练功能中统计输出量化信息模块，和预测端打通，提升鲁棒性。 ([#31680](https://github.com/PaddlePaddle/Paddle/pull/31680), [#31710](https://github.com/PaddlePaddle/Paddle/pull/31710), [#31861](https://github.com/PaddlePaddle/Paddle/pull/31861))
    - 动态图量化训练支持和混合精度训练结合使用。([#33484](https://github.com/PaddlePaddle/Paddle/pull/33484))
    - 动态图量化训练功能支持对Function类API进行量化。([#33162](https://github.com/PaddlePaddle/Paddle/pull/33162), [#33871](https://github.com/PaddlePaddle/Paddle/pull/33871))
    - 支持静态图模式下分布式量化训练。 ([#33781](https://github.com/PaddlePaddle/Paddle/pull/33781))
    - 支持动态图模式下conv2d_transpose的量化。([#34547](https://github.com/PaddlePaddle/Paddle/pull/34547))

- 自定义OP
    - 新增自定义算子 DCU 后端支持。([#34050](https://github.com/PaddlePaddle/Paddle/pull/34050))
	
- Cost Model
    - 新增 Paddle CostModel，实现通过 Profiler 获取 op 时间 cost 的方法。 ([#35774](https://github.com/PaddlePaddle/Paddle/pull/35774)) 

- 模型保存与载入 
    - 新增通过 ``paddle.jit.save`` 接口直接将 Layer 的非 forward 成员方法及相关参数保存为推理模型的功能。 ([#34070](https://github.com/PaddlePaddle/Paddle/pull/34070))


- ONNX Exporter 
    - 新增8个算子适配： `softplus`、`elementwise_mod`、 `elementwise_floordiv`、`p_norm`、`depthwise_transpose`、`group_norm`、`pixel_shuffle`、`top_k`。([Paddle2ONNX#252](https://github.com/PaddlePaddle/Paddle2ONNX/pull/252),  [Paddle2ONNX#261](https://github.com/PaddlePaddle/Paddle2ONNX/pull/261),  [Paddle2ONNX#293](https://github.com/PaddlePaddle/Paddle2ONNX/pull/293))
    - 新增8个检测模型导出：PPYOLO、PPYOLOv2、PPYOLO-Tiny、TTFNet、PAFNet、FCOS、SSD。 ([Paddle2ONNX#252](https://github.com/PaddlePaddle/Paddle2ONNX/pull/252))

### （2）功能优化

#### API
- `paddle.slice` 增加对`bool`类型Tensor的支持以及优化了报错信息。([#35586](https://github.com/PaddlePaddle/Paddle/pull/35586), [#35179](https://github.com/PaddlePaddle/Paddle/pull/35179))
- `paddle.strided_slice`新增对`TensorArray`类型输入的支持，调整了`step<0`时的输出结果，调整后的结果与`numpy`保持一致。([#34205](https://github.com/PaddlePaddle/Paddle/pull/34205), [#34172](https://github.com/PaddlePaddle/Paddle/pull/34172))
- ``paddle.multiply`` 支持 ``bool`` 数据类型的运算。([#35551](https://github.com/PaddlePaddle/Paddle/pull/35551))
- 逻辑运算(``paddle.logical_not, paddle.logical_and, paddle.logical_or, paddle.logical_xor``)支持非 ``bool`` 数据类型(``int8, int16, int32, int64, float, double``)。([#34141](https://github.com/PaddlePaddle/Paddle/pull/34141))
- ``paddle.transpose`` 支持 ``bool`` 类型运算。([#35886](https://github.com/PaddlePaddle/Paddle/pull/35886))
- ``paddle.strided_slice`` 支持 ``bool`` 类型运算。([#33373](https://github.com/PaddlePaddle/Paddle/pull/33373))
- ``paddle.set_printoptions`` 支持设置 ``linewidth`` 来打印 ``Tensor`` 。([#35175](https://github.com/PaddlePaddle/Paddle/pull/35175))
- ``paddle.to_tensor`` 支持 ``LoDTensor`` 。([#33027](https://github.com/PaddlePaddle/Paddle/pull/33027))
- ``paddle.linalg.det`` 和 ``paddle.linalg.slogdet`` 支持反向运算。([#36013](https://github.com/PaddlePaddle/Paddle/pull/36013))
- ``paddle.nn.functional.pad`` 支持全维度pad时，tuple类型pad参数的输入。 ([35985](https://github.com/PaddlePaddle/Paddle/pull/35985))
- 优化``paddle.nn.functional.pad`` 输入异常时的报错信息。 ([34979](https://github.com/PaddlePaddle/Paddle/pull/34979))
- 静态图支持对部分 ``program``，生成相应的反向``program``。([#34395](https://github.com/PaddlePaddle/Paddle/pull/34395))
- oneDNN 功能优化
    - 新增多个算子的oneDNN kernels支持，包括新增对 ``clip``、``slice``、``split``、``cast``、 ``scale``、``expand_v2``、``sigmoid``、``matmul_v2``、``PRelu`` 的前向和反向 oneDNN FP32 和 oneNheN BF16的支持。([#35601](https://github.com/PaddlePaddle/Paddle/pull/35601), [#34332](https://github.com/PaddlePaddle/Paddle/pull/34332), [#34284](https://github.com/PaddlePaddle/Paddle/pull/34284), [#34216](https://github.com/PaddlePaddle/Paddle/pull/34216), [#34192](https://github.com/PaddlePaddle/Paddle/pull/34192),  [#33878](https://github.com/PaddlePaddle/Paddle/pull/33878), [#33584](https://github.com/PaddlePaddle/Paddle/pull/33584), [#33056](https://github.com/PaddlePaddle/Paddle/pull/33056), [#32975](https://github.com/PaddlePaddle/Paddle/pull/32975))
  - 新增SGD算子中 Selected rows 使用 oneDNN AXPY 的实现。([33632](https://github.com/PaddlePaddle/Paddle/pull/33632))
- Ampere 架构的GPU上支持 ``bfloat16`` 数据类型。([#31232](https://github.com/PaddlePaddle/Paddle/pull/32132), [#32221](https://github.com/PaddlePaddle/Paddle/pull/32221), [#32542](https://github.com/PaddlePaddle/Paddle/pull/32542))
- Ampere 架构的GPU上 ``Conv`` 算子设置使用 Tensor Core 。([#34409](https://github.com/PaddlePaddle/Paddle/pull/34409))
- 支持 ``paddle.device.cuda.current_stream().cuda_stream`` 获取裸指针。([#35813](https://github.com/PaddlePaddle/Paddle/pull/35813))
- 新增``paddle.optimizer.AdamW`` GPU fuse kernel 实现，并支持 layerwise learning rate 功能。([#35020](https://github.com/PaddlePaddle/Paddle/pull/35020), [#35569](https://github.com/PaddlePaddle/Paddle/pull/35569))
- 支持在 paddle 中使用Nvidia的cusparse库函数。([#35675](https://github.com/PaddlePaddle/Paddle/pull/35675))
- 新增 ``paddle.full`` 对 ``int16`` 类型的支持。([#35619](https://github.com/PaddlePaddle/Paddle/pull/35619))
- 优化 ``paddle.nn.ClipGradByGlobalNorm`` 的显存占用。([#34586](https://github.com/PaddlePaddle/Paddle/pull/34586))
- `reduce_sum` 算子支持float16类型([#32966](https://github.com/PaddlePaddle/Paddle/pull/32966))
- `paddle.nn.CTCLoss` 新增两种 grad norm 方法`norm_by_total_logits_len` 和 `norm_by_batchsize` 。([#34729](https://github.com/PaddlePaddle/Paddle/pull/34729/)) 
- 新增各路径下公开API推荐使用路径。([#33313](https://github.com/PaddlePaddle/Paddle/pull/33313), [#33308](https://github.com/PaddlePaddle/Paddle/pull/33308), [#32759](https://github.com/PaddlePaddle/Paddle/pull/32759), [#32695](https://github.com/PaddlePaddle/Paddle/pull/32695), [#32643](https://github.com/PaddlePaddle/Paddle/pull/32643), [#31912](https://github.com/PaddlePaddle/Paddle/pull/31912), [#32650](https://github.com/PaddlePaddle/Paddle/pull/32650), [#32034](https://github.com/PaddlePaddle/Paddle/pull/32034), [#33897](https://github.com/PaddlePaddle/Paddle/pull/33897)) 
- 恢复 `paddle.vision` 路径下原API可访问性。([#34432](https://github.com/PaddlePaddle/Paddle/pull/34432))
-  `paddle.vision.ops.deform_conv2d, paddle.vision.ops.DeformConv2D` 新增 double 输入类型支持。 ([#35330](https://github.com/PaddlePaddle/Paddle/pull/35330))
- `paddle.fluid.contrib.layers.shuffle_batch` 新增 GPU Kernel实现。[#33938](https://github.com/PaddlePaddle/Paddle/pull/33938) 
- 已有API新增公开调用路径 `paddle.linalg.cholesky`, `paddle.linalg.norm`, `paddle.linalg.inv`。([#33420](https://github.com/PaddlePaddle/Paddle/pull/33420)) 
- `paddle.reshape` 支持将空 `Tensor` 形变成另一个形状的空 `Tensor`。([#36087](https://github.com/PaddlePaddle/Paddle/pull/36087))
- `paddle.equal`第二个输入新增 `int`、`float` 和 `bool` 类型的支持。([#35695](https://github.com/PaddlePaddle/Paddle/pull/35695))
- ``paddle.io.DataLoader``新增支持persistent_worker模式。([#34017](https://github.com/PaddlePaddle/Paddle/pull/34017))
- 优化``l2_normalize``,``p_norm``,``elementwise_max``,``prelu``,``clip_by_norm``,``lars optimizer``算子支持float16计算。 ([#35576](https://github.com/PaddlePaddle/Paddle/pull/35576), [#35888](https://github.com/PaddlePaddle/Paddle/pull/35888), [#35888](https://github.com/PaddlePaddle/Paddle/pull/35888), [35532](https://github.com/PaddlePaddle/Paddle/pull/35532), [#35446](https://github.com/PaddlePaddle/Paddle/pull/35446), [#33280](https://github.com/PaddlePaddle/Paddle/pull/33280))
- 优化flowers数据集的读取速度，从每批次数分钟优化至1~3秒。([#31408](https://github.com/PaddlePaddle/Paddle/pull/31408))
- 支持`paddle.distributed.fleet.DistributedStrategy` 中 `without_graph_optimize` 开关打开后的fuse allreduce sum功能。FP32下性能提升3%，AMP下性能提升8%。([#34446](https://github.com/PaddlePaddle/Paddle/pull/34446)) 
- `paddle.matmul` 将底层Op算子由matmul op 切换到 matmul_v2 op。 ([#36374](https://github.com/PaddlePaddle/Paddle/pull/36374))
- `paddle.fft` 模块添加了 mkl_cdft 和 hipfft 两个计算后端。 ([#36537](https://github.com/PaddlePaddle/Paddle/pull/36537))
- `paddle.roll` 的参数 `shifts` 支持 `Tensor` 作为输入。 ([#36537](https://github.com/PaddlePaddle/Paddle/pull/36537))
- `paddle.shape` 支持复数类型的输入。([#36835](https://github.com/PaddlePaddle/Paddle/pull/36835))
- matmul_v2 支持量化。([#36469](https://github.com/PaddlePaddle/Paddle/pull/36469))
- 新增 `clip_op` 对 `float16` 的支持。 ([#36672](https://github.com/PaddlePaddle/Paddle/pull/36672))
- `paddle.fft` 模块为 cufft 后端添加了缓存 plan 的功能，优化性能。([#36537](https://github.com/PaddlePaddle/Paddle/pull/36537))

#### IR(Intermediate Representation)
- 动态图转静态图
    - 优化动转静报错格式，隐藏框架层不必要的报错栈，添加用户代码报错行定位标识和上下文。([#35365](https://github.com/PaddlePaddle/Paddle/pull/35365), [#35320](https://github.com/PaddlePaddle/Paddle/pull/35320))
    - 优化控制流中 ``list.append`` 语法的转换逻辑。([#35212](https://github.com/PaddlePaddle/Paddle/pull/35212)) 
    - 优化了动转静训练代码逻辑，升级内部 ``Program`` 缓存机制，新增输入 ``Tensor`` 的提前 copy 策略，提升训练性能。 ([#34181](https://github.com/PaddlePaddle/Paddle/pull/34181), [#33796](https://github.com/PaddlePaddle/Paddle/pull/33796))
    - 优化动转静内部执行器显存回收策略，减少训练时显存占用量。 ([#34177](https://github.com/PaddlePaddle/Paddle/pull/34177))
    - 集成了 ``Gast`` 三方依赖库的源码，解耦了版本依赖。 ([#34556](https://github.com/PaddlePaddle/Paddle/pull/34556)) 
    - 动转静报错时显示部分框架层报错信息，使得定位问题更加容易。([#36765](https://github.com/PaddlePaddle/Paddle/pull/36765))
    - 移除动转静报错模块中重复的临时文件删除函数`remove_static_file()`。([#36375](https://github.com/PaddlePaddle/Paddle/pull/36375))
    - 优化对RegisterPass中`input_specs`参数处理，支持图优化时作为匹配子图条件。([#36453](https://github.com/PaddlePaddle/Paddle/pull/36453))


#### 分布式训练
- 分布式训练基础功能
    - 增强静态图流水线并行 stage 以及 persist var 的检查。([#34193](https://github.com/PaddlePaddle/Paddle/pull/34193), [#34870](https://github.com/PaddlePaddle/Paddle/pull/34870), [#35453](https://github.com/PaddlePaddle/Paddle/pull/35453))
    - 优化静态图流水线并行，1F1B 调度使显存不随 global batch size 增大而增大。([#34230](https://github.com/PaddlePaddle/Paddle/pull/34230))
    - GPU 参数服务器优化构建阶段 hashmap，构建阶段性能在某些任务上提升可达7倍。([#34175](https://github.com/PaddlePaddle/Paddle/pull/34175)) 
    - GPU 参数服务器 pull/push 阶段新增多流并行。([#34276](https://github.com/PaddlePaddle/Paddle/pull/34276)) 
    - GPU 参数服务器支持多机训练模式下，机器间远程拉取参数。([#35396](https://github.com/PaddlePaddle/Paddle/pull/35396))
    - CPU 参数服务器支持 SSD存储。 ([#33031](https://github.com/PaddlePaddle/Paddle/pull/33031))
    - `paddle.io.Dataset` 支持动态库解析数据。 ([#33969](https://github.com/PaddlePaddle/Paddle/pull/33969))
    - 新增 `paddle.distributed.fleet.dataset.DatasetBase` 中对`use_var_list`和 `pipe_command` 生成数据的一致性检查函数。 ([#34463](https://github.com/PaddlePaddle/Paddle/pull/34463))
    - 新增 `paddle.fluid.layers.embedding` 的 `emd` 维度与 `fleet` 中` sparse table` 的 `emb` 维度的一致性检查。 ([#34249](https://github.com/PaddlePaddle/Paddle/pull/34249))
    - 动态图混合并行支持Pure FP16训练。([#36707](https://github.com/PaddlePaddle/Paddle/pull/36707))
    - 静态图混合并行支持dropout使用固定随机种子生成器，以确保模型并行中全局变量的一致性与局部变量的随机性。([#36682](https://github.com/PaddlePaddle/Paddle/pull/36682))
    ‘
    - 实现了CPU并行，并支持调用 spawn 或 launch 时可以添加自定义的backend参数。可用的backend选择为 "gloo", "nccl", "bkcl", "auto" ，分别表示CPU并行，GPU并行，XPU并行和按照Paddle版本自动选择。([#35745](https://github.com/PaddlePaddle/Paddle/pull/35745))
    - 优化动态图混合并行 HybridParallelClipGrad 策略，支持4D混合并行+Pure FP16训练。([#36707](https://github.com/PaddlePaddle/Paddle/pull/36707))
    - 添加 SlotRecordDataset 类支持GPU参数服务器训练。([#36710](https://github.com/PaddlePaddle/Paddle/pull/36710))
    - GPU参数服务器构建阶段支持使用SlotRecordDataset。([#36723](https://github.com/PaddlePaddle/Paddle/pull/36723))

- 静态图混合并行
    - 优化混合并行 loss scale，减少 scale op 插入个数。([#35775](https://github.com/PaddlePaddle/Paddle/pull/35775))
    - 优化 pipeline 的调度器，cache 重复的 cpu 工作，降低 cpu 开销。([#35680](https://github.com/PaddlePaddle/Paddle/pull/35680))
    - 优化流水线并行 + recompute 时 checkpoint send/recv 的次数。([#34248](https://github.com/PaddlePaddle/Paddle/pull/34248))


#### 其他
- 报错调试优化
    - 统一第三方库报错信息机制，优化 ``CURAND、CUDNN、CUBLAS、CUSOLVER、NCCL`` 五种 CUDA API 的报错信息，使报错内容更加详细与规范。 ([#33003](https://github.com/PaddlePaddle/Paddle/pull/33003), [#33743](https://github.com/PaddlePaddle/Paddle/pull/33743))
    - 优化 avx 与 no_avx 相关的安装报错信息，简化冗余复杂内容。 ([#33818](https://github.com/PaddlePaddle/Paddle/pull/33818)) 
    - 优化 ``paddle.nn.functional.gather_tree``，``paddle.nn.Transformer``，``paddle.nn.TransformerDecoderLayer``，``paddle.nn.TransformerEncoderLayer``，``paddle.nn.MultiHeadAttention`` 报错信息。 ([#34322](https://github.com/PaddlePaddle/Paddle/pull/34322), [#33859](https://github.com/PaddlePaddle/Paddle/pull/33859))
    - 支持在动态图下配置 ``FLAGS_check_nan_inf``环境变量， 用于模型 ``nan`` 和 ``inf`` 的运行时检查与定位。 ([#32635](https://github.com/PaddlePaddle/Paddle/pull/32635))
    - 移除 Signal 类报错信息中由于捕获 Signal 引入的栈信息，避免误导用户。([#34842 ](https://github.com/PaddlePaddle/Paddle/pull/34842))
    - 修复 ``elementwise`` 类算子在输入x或y为空 Tensor 时的报错信息。 ([#33928](https://github.com/PaddlePaddle/Paddle/pull/33928))

- 模型保存与载入
	- 修正 ``paddle.jit.save``  接口和模型裁剪的逻辑，不再为输出变量增加一个关联的 ``scale_op``，可以正确导出含有 ``bool``，``float16`` 类型输出的模型。([#35730](https://github.com/PaddlePaddle/Paddle/pull/35730), [#36132](https://github.com/PaddlePaddle/Paddle/pull/36132))
- 自定义OP
	- 移除 ``paddle::Tensor`` 的 ``copy`` 方法中不必要的 ``cudaStreamSynchronize`` 操作，以提升性能。([#35802](https://github.com/PaddlePaddle/Paddle/pull/35802))
- 新增C++对GeneratePass开发注册的支持，开发方式与Python侧对齐。([#36302](https://github.com/PaddlePaddle/Paddle/pull/36302))
- 自动稀疏化训练(Automic SParsity)
	- 新增`paddle.static.sparsity`，支持生成`n:m`稀疏模式的稀疏参数，目前只支持静态图ASP训练。A100上FP32、FP16分别设置`1:2`、`2:4`的稀疏模式，训练保存的稀疏模型，可通过调用TensorRT 8利用Ampere架构的稀疏Tensor Core加速推理任务。当前版本共提供了5个API：([#32995](https://github.com/PaddlePaddle/Paddle/pull/32995)、[#33132](https://github.com/PaddlePaddle/Paddle/pull/33132)、[#33558](https://github.com/PaddlePaddle/Paddle/pull/33558)、[#36525](https://github.com/PaddlePaddle/Paddle/pull/36525))
	- `paddle.static.sparsity.calculate_density`，计算输入Tensor的密度。
	- `paddle.static.sparsity.decorate`，将给定的优化器包装为`OptimizerWithSparsityGuarantee`，在调用 `optimizer.minimize()`时自动为ASP工作流插入必要的操作。
	- `paddle.static.sparsity.prune_model`，依据`mask_algo`指定的掩码生成函数裁剪`main_program`中支持的层的参数。
	- `paddle.static.sparsity.set_excluded_layers`，设置不会被裁剪的层的参数名称。
	- `paddle.static.sparsity.reset_excluded_layers`，重置与`main_program`相对应的`excluded_layers`设置。



### （3）性能优化

#### 分布式训练-静态图混合并行

- 优化模型并行 + AMP 时 AMP 的灰名单列表，支持模型并行算子，性能提升8%。([#33660](https://github.com/PaddlePaddle/Paddle/pull/33660))
- 优化流水线并行时反向梯度累加的 `device` 属性设置，性能提升1-3%。([#33946](https://github.com/PaddlePaddle/Paddle/pull/33946))
- 优化流水线并行执行器中 debug 的部分，性能提升60-140%。 ([#33948](https://gifothub.com/PaddlePaddle/Paddle/pull/33948))
- 支持流水线并行下 `Program` cache的功能，性能提升10-40%。([#33998](https://github.com/PaddlePaddle/Paddle/pull/33998), [#33954](https://github.com/PaddlePaddle/Paddle/pull/33954))
- 优化流水线并行 `send` 的通信等待，性能提升0.3-2%。([#34086](https://github.com/PaddlePaddle/Paddle/pull/34086)) 
- 优化模型并行 + 流水线并行时 `send/recv`发送数据量的大小，8机测试性能提升36%。([#34110](https://github.com/PaddlePaddle/Paddle/pull/34110))
- 优化混合并行 + AMP时参数的 cast，通过`optimize_cast`控制，性能可提升5-7%。([#34965](https://github.com/PaddlePaddle/Paddle/pull/34965))
- 优化流水线并行 + recompute + amp 时的性能，性能提升13%。([#34519](https://github.com/PaddlePaddle/Paddle/pull/34519))
- 支持流水线并行 + 数据并行时使用``float16``通信，通过`distributed_strategy.fp16_allreduce`控制，性能可提升13%。([#34762](https://github.com/PaddlePaddle/Paddle/pull/34762))

#### 算子优化

- 设计并实现了通用的Reduce CUDA算法，应用于7个Reduce算子，加速1.0x ~ 22.7x。([#32697](https://github.com/PaddlePaddle/Paddle/pull/32697), [#32974](https://github.com/PaddlePaddle/Paddle/pull/32974), [#33267](https://github.com/PaddlePaddle/Paddle/pull/33267), [#32885](https://github.com/PaddlePaddle/Paddle/pull/32885), [#33144](https://github.com/PaddlePaddle/Paddle/pull/33144),  [#33761](https://github.com/PaddlePaddle/Paddle/pull/33761), [#33901](https://github.com/PaddlePaddle/Paddle/pull/33901), [#34143](https://github.com/PaddlePaddle/Paddle/pull/34143),  [#34436](https://github.com/PaddlePaddle/Paddle/pull/34436))
- 设计并实现了通用的Elementwise和Broadcast CUDA算法。([#32512](https://github.com/PaddlePaddle/Paddle/pull/32512), [#32928](https://github.com/PaddlePaddle/Paddle/pull/32928), [#33976](https://github.com/PaddlePaddle/Paddle/pull/33976), [#32148](https://github.com/PaddlePaddle/Paddle/pull/32148), [#32414](https://github.com/PaddlePaddle/Paddle/pull/32414))：应用于41个一元、激活算子。([#32348](https://github.com/PaddlePaddle/Paddle/pull/32348), [#32622](https://github.com/PaddlePaddle/Paddle/pull/32622), [#32823](https://github.com/PaddlePaddle/Paddle/pull/32823))，性能提升1.1x ~ 1.4x；应用于19个二元（9个基础计算类、6个比较类、4个逻辑类）算子。([#33050](https://github.com/PaddlePaddle/Paddle/pull/33050), [33052](https://github.com/PaddlePaddle/Paddle/pull/33052), [#33053](https://github.com/PaddlePaddle/Paddle/pull/33053), [#33051](https://github.com/PaddlePaddle/Paddle/pull/33051), [#33089](https://github.com/PaddlePaddle/Paddle/pull/33089))，性能提升1.02x ~ 3.21x。
- 优化``roll``算子CUDA实现 ，单维度、多维度输入时，性能分别提升10%、50%以上。([#32880](https://github.com/PaddlePaddle/Paddle/pull/32880))
- 优化``roll``算子index计算，单维度、多维度性能分别提升15%和70%。([#33909](https://github.com/PaddlePaddle/Paddle/pull/33909))
- 优化`update_loss_scaling_op`算子CUDA实现，性能提升2.06x。([#32554](https://github.com/PaddlePaddle/Paddle/pull/32554))
- 优化 ``softmax_with_cross_entropy (hard label)`` GPU 算子性能，加速比1.0x ~ 10.0x。([#35660](https://github.com/PaddlePaddle/Paddle/pull/35660))
- 优化``index_select``前、反向算子的CPU实现，加速比达到2.09x~9.34x。([#32863](https://github.com/PaddlePaddle/Paddle/pull/32863),  [#32955](https://github.com/PaddlePaddle/Paddle/pull/32955))
- 优化``batch_norm``算子二维输入情况下的CPU实现，提升达到22.68x~30.00x。([#34585](https://github.com/PaddlePaddle/Paddle/pull/34585))
- 优化``batch_norm``算子在初始化方式及二维输入下的GPU性能，提升1.25x~25x。([#33851](https://github.com/PaddlePaddle/Paddle/pull/33851), [#33887](https://github.com/PaddlePaddle/Paddle/pull/33887))
- ``log_softmax``算子性能优化及该相关bug修复，优化后较优化前kernel性能对比4.22x~32.29x。 ([#31630](https://github.com/PaddlePaddle/Paddle/pull/31630), [#32180](https://github.com/PaddlePaddle/Paddle/pull/32180), [#32396](https://github.com/PaddlePaddle/Paddle/pull/32396), [#32937](https://github.com/PaddlePaddle/Paddle/pull/32937))
- 优化``concat_and_split``算子，解决动态图在海光DCU芯片上训练BERT时计算和通信无法overlap的问题，在海光DCU芯片上BERT分布式训练性能提升约27%。([#33982](https://github.com/PaddlePaddle/Paddle/pull/33982))
- 优化``fused_elemwise_act``算子，MB计算规模下有十余倍性能提升。([#33480](https://github.com/PaddlePaddle/Paddle/pull/33480))
	
#### 策略优化

- 增加``build_strategy.fix_op_run_order``策略，固定op执行的次序，ResNet模型单机8卡速度提升1.8%。([#34427](https://github.com/PaddlePaddle/Paddle/pull/34427))
- 动态图反向计算支持并自动开启部分算子inplace策略，动态图gpt模型pure float16训练性能提升4.8%。 ([#35412](https://github.com/PaddlePaddle/Paddle/pull/35412))
- 优化动态图性能，将只在静态图执行的逻辑从动态图的执行路径中剥离。([#34024](https://github.com/PaddlePaddle/Paddle/pull/34024))
- IR Pass优化能力作为通用能力露出，同时支持单机和分布式优化。在GPT混合并行场景性能提升3%-5%。([#34955](https://github.com/PaddlePaddle/Paddle/pull/34955), [#35704](https://github.com/PaddlePaddle/Paddle/pull/35704), [#34730](https://github.com/PaddlePaddle/Paddle/pull/34730), [#34524](https://github.com/PaddlePaddle/Paddle/pull/34524))
- 优化 ctc loss grad 计算速度，提速~3x，但相应增加了GPU显存占用。([#34729](https://github.com/PaddlePadle/Paddle/pull/34729))
- transformer encoder 性能优化
	- 优化思路：通过新增 `paddle.incubate.nn.FusedMultiHeadAttention` 和 `paddle.incubate.nn.FusedFeedForward` 的方式，在实现中采用 q, k, v gemm融合及多种kernel融合优化技术，提升transformer encoder的性能。
		- FusedAttention
			- 新增 `paddle.incubate.nn.functional.fused_multi_head_attention` ，支持multi-head attention的融合计算。([#35905](https://github.com/PaddlePaddle/Paddle/pull/35905) [35903](https://github.com/PaddlePaddle/Paddle/pull/35903) [#36803](https://github.com/PaddlePaddle/Paddle/pull/36803) [#36793](https://github.com/PaddlePaddle/Paddle/pull/36793) [36185](https://github.com/PaddlePaddle/Paddle/pull/36185))
			- 新增 `paddle.incubate.nn.FusedMultiHeadAttention` ，用于融合multi-head attention的layer层组网。 ([#36498](https://github.com/PaddlePaddle/Paddle/pull/36498) )
			- 该模块使用q, k, v gemm融合和bias add + dropout + residual add + layer_norm kernel融合优化技术，可带来1.08x-1.45x加速。
		
		- FusedFeedForward
			- 新增 `paddle.incubate.nn.functional.fused_feedforward` ，支持 feedforward的融合计算。([#36729](https://github.com/PaddlePaddle/Paddle/pull/36729) [#36730](https://github.com/PaddlePaddle/Paddle/pull/36730))
			- 新增 `paddle.incubate.nn.FusedFeedForward` ，用于融合feedforward的layer层组网。 ([#36776](https://github.com/PaddlePaddle/Paddle/pull/36776))
			- 性能较优化前有1.04x~1.22x左右的提升。
		- 新增 `paddle.incubate.nn.FusedTransformerEncoderLayer`，支持使用融合multi-head attention和融合feedforward计算的layer层组网。 ([#36776](https://github.com/PaddlePaddle/Paddle/pull/36776))


### （4）问题修复

#### API

-  优化`depthwise_conv` 数值稳定性。 ([#35161](https://github.com/PaddlePaddle/Paddle/pull/35161))
- 添加参数创建时的形状检查，以保证参数每个轴的 `size` 都大于 0 。([#33265](https://github.com/PaddlePaddle/Paddle/pull/33265))
- 优化``paddle.nn.LayerNorm``的计算，并修复数据溢出相关bug。([#34432](https://github.com/PaddlePaddle/Paddle/pull/34432), [#33658](https://github.com/PaddlePaddle/Paddle/pull/33658))
- 支持Windows应用场景，将PaddlePaddle 框架能力集成到 MFC/QT/C# 等桌面端软件环境中，修复进程嵌套导致系统崩溃问题。([#34312](https://github.com/PaddlePaddle/Paddle/pull/34312))
- 修复Reduce 数据初始化导致NLP 模型loss有误的问题。([#34941](https://github.com/PaddlePaddle/Paddle/pull/34941))
- 修复``paddle.nn.LayerNorm``在`batch_size=1`时候的bug问题。([#35480](https://github.com/PaddlePaddle/Paddle/pull/35480))
- 修复``paddle.static.nn.group_norm``在空输入下不能正确捕获错误的问题。([#35613](https://github.com/PaddlePaddle/Paddle/pull/35613))
- 修复``paddle.nn.functional.batch_norm``在 `is_test=True` 的情况下mean/variance为空的问题。([#35328](https://github.com/PaddlePaddle/Paddle/pull/35328))
- 修复``paddle.nn.functional.instance_norm``和``paddle.nn.functional.batch_norm``输入为空时，访问越界的问题。([#35341](https://github.com/PaddlePaddle/Paddle/pull/35341), [#34107](https://github.com/PaddlePaddle/Paddle/pull/34107))
- 修复量化模型不统计``paddle.nn.LayerNorm`` 的输出的问题。([#33610](https://github.com/PaddlePaddle/Paddle/pull/33610))
- 修复``paddle.nn.SyncBatchNorm.convert_sync_batchnorm()``不支持1D/3D的问题 。([#32989](https://github.com/PaddlePaddle/Paddle/pull/32989))
- 修复``paddle.nn.BatchNorm1D, paddle.nn.BatchNorm2D, paddle.nn.BatchNorm3D``在`is_test=True`的情况下无法添加反向的问题。([#32678](https://github.com/PaddlePaddle/Paddle/pull/32678))
- 修复`Tensor.cuda`不支持`device_id`为`None`的问题。 ([#34416](https://github.com/PaddlePaddle/Paddle/pull/34416))
- 修复``paddle.to_tensor``不支持 `Tensor.dtype, core.Tensor`等内置类型的问题。 ([#31931](https://github.com/PaddlePaddle/Paddle/pull/31931), [#33430](https://github.com/PaddlePaddle/Paddle/pull/33430))
- 修复`paddle.nn.functional.log_softmax`不支持输入维度为0的问题。([#34635](https://github.com/PaddlePaddle/Paddle/pull/34635))
- 修复``paddle.nn.GroupNorm`` 在float32下CPU计算结果和准确值的相对误差大于1e-5的问题。([#33176](https://github.com/PaddlePaddle/Paddle/pull/33176))
- 修复``paddle.trace`` 在参数 `offset` 超出维度大小时返回结果不为0的问题，在参数 `axis1`和`axis2` 输入不合法值时的栈溢出问题。([#33922](https://github.com/PaddlePaddle/Paddle/pull/33922), [#35419](https://github.com/PaddlePaddle/Paddle/pull/35419))
- 修复``paddle.sum``输入参数为bool类型时，输出类型不为int的问题。输入参数类型和输出参数类型不一致且 axis 轴对应的reduce元素个数为1时，输出类型错误问题。([#34313](https://github.com/PaddlePaddle/Paddle/pull/34313), [#36123](https://github.com/PaddlePaddle/Paddle/pull/36123))
- 修复 ``paddle.nn.conv2d/conv3d/conv2d_transpose/conv3d_transpose``非法输入时除0错误和数组越界的问题。([#35337](https://github.com/PaddlePaddle/Paddle/pull/35337))
- 修复``paddle.nn.conv2d_transpose``非法输入时堆缓冲区溢出的问题。([#35340](https://github.com/PaddlePaddle/Paddle/pull/35340))
- 修复 ``paddle.bmm`` 写空地址导致程序运行时崩溃的问题。([#35098](https://github.com/PaddlePaddle/Paddle/pull/35098))
- 修复 ``cast`` 算子无法支持 Tensor 从int16 转换到float32的问题。([#35156](https://github.com/PaddlePaddle/Paddle/pull/35156))
-  修复 `assign` 不支持float16和uint8的问题。([#35153](https://github.com/PaddlePaddle/Paddle/pull/35153))
-  修复 `concat` 在输入大shape tensor时，容易溢出的问题。([#34319](https://github.com/PaddlePaddle/Paddle/pull/34319))
- 修复动态图 `concat` 不支持空tensor作为输入的问题。([#35845](https://github.com/PaddlePaddle/Paddle/pull/35845))
- 修复 ``paddle.where``不支持broadcast的问题。([#35092](https://github.com/PaddlePaddle/Paddle/pull/35092))
- 修复 ``paddle.reshape`` 空tensor 时输入合法性未检查问题。([#35642](https://github.com/PaddlePaddle/Paddle/pull/35642))
- 修复 ``layernorm`` 算子在大shape下cuda kernel配错错误问题。 ( [#33748](https://github.com/PaddlePaddle/Paddle/pull/33748))
- 修复 ``random``类算子静态图下stop_gradient属性设置错误问题。( [#33959](https://github.com/PaddlePaddle/Paddle/pull/33959))
- 修复 ``split`` 算子输入为空tensor的错误行为。([#334356](https://github.com/PaddlePaddle/Paddle/pull/334356))
- 修复 tensor 的 slice 左值赋值显存泄漏问题。([#35013](https://github.com/PaddlePaddle/Paddle/pull/35013))
- 修复动态图Layer无法被cloudpickle dump和load的问题。([#35538](https://github.com/PaddlePaddle/Paddle/pull/35538))
- 修复simple_rnn_cell, gru_cell和lstm_cell API 非法参数设置导致除零错误问题。([#34627](https://github.com/PaddlePaddle/Paddle/pull/34627))
- 修复``paddle.nn.functional.linear``在非法输入时空指针解引用的问题。([#34696](https://github.com/PaddlePaddle/Paddle/pull/34696))
- 修复``paddle.strided_slice``,``paddle.transpose``存在内存越界问题。([#35062](https://github.com/PaddlePaddle/Paddle/pull/35062), [#35079](https://github.com/PaddlePaddle/Paddle/pull/35079))
- 修复``roll``算子非法输入时除0错误的问题。([#34499](https://github.com/PaddlePaddle/Paddle/pull/34499))
- 修复``gather``算子非法输入时的数组越界问题。([#34096](https://github.com/PaddlePaddle/Paddle/pull/34096), [#34138](https://github.com/PaddlePaddle/Paddle/pull/34138), [#34200](https://github.com/PaddlePaddle/Paddle/pull/34200))
- 修复``prelu``，``softlax``算子非法输入时除0错误的问题。([#34499](https://github.com/PaddlePaddle/Paddle/pull/34499))
- 修复``split``算子未对输入参数做合法性检查问题。([#34630](https://github.com/PaddlePaddle/Paddle/pull/34630))
- 修复``memcpy``算子无法支持海光DCU芯片的问题。([#35394](https://github.com/PaddlePaddle/Paddle/pull/35394))
- 修复``slice``算子在`batch_size=1`下训练会报错问题。([#34265](https://github.com/PaddlePaddle/Paddle/pull/34265))
- 修复``reduce_sum``算子在 AMP 下容易溢出问题。([#33960](https://github.com/PaddlePaddle/Paddle/pull/33960))
- 修复ANSI转义代码在windows下显示错乱问题。([#33689](https://github.com/PaddlePaddle/Paddle/pull/33689))
- 修复``paddle.hub``解析文件名字和下载保存文件不一致问题。([#33214](https://github.com/PaddlePaddle/Paddle/pull/33214))
- 修复``matmul``, ``diag_embed``, `` auc ``算子输入空tensor时内存泄露问题。 ([#34978](https://github.com/PaddlePaddle/Paddle/pull/34978))
- 修复 ``paddle.less_equal, paddle.less_than, paddle.greater_equal, paddle.greater_than`` 计算broadcast计算精度误差大的BUG。([#32941](https://github.com/PaddlePaddle/Paddle/pull/32941))
- 修复 ``interpolate`` 算子在大输入shape下的崩溃问题。([#35577](https://github.com/PaddlePaddle/Paddle/pull/35577))
- 修复 ``interpolate``, ``unfold``, ``spectral_norm`` 算子输入为空tensor的合法性检查问题。([#33941](https://github.com/PaddlePaddle/Paddle/pull/33941), [#34943](https://github.com/PaddlePaddle/Paddle/pull/34943), [#35005](https://github.com/PaddlePaddle/Paddle/pull/35005))
- 修复`paddle.flops`在计算输出的FLOPs可能出现负号（整数溢出）的问题。([#33576](https://github.com/PaddlePaddle/Paddle/pull/33576))
- 修复``paddle.summary``遇到返回值含非Tensor元素的层时报错的问题。([#34160](https://github.com/PaddlePaddle/Paddle/pull/34160))
- 修复``pool``算子非法输入时计算输出shape错误的问题。([#35106](https://github.com/PaddlePaddle/Paddle/pull/35106))
- 修复 ``unfold, dice_loss, reshape``算子输入shape的合法性检查问题。([#34673](https://github.com/PaddlePaddle/Paddle/pull/34673), [#34757](https://github.com/PaddlePaddle/Paddle/pull/34757), [#35016](https://github.com/PaddlePaddle/Paddle/pull/35016))
- 修复``unique, unstack``算子输入zero tensor的问题。([#36021](https://github.com/PaddlePaddle/Paddle/pull/36021))
- 修复stack算子的反向输入为空时的问题。([#362877](https://github.com/PaddlePaddle/Paddle/pull/32877))
- 修复 ``paddle.inverse``在输入Tensor的形状为`[0, 0, 0]`时，CPU执行会出现除0错误的问题。([#34996](https://github.com/PaddlePaddle/Paddle/pull/34996))
- 修复``paddle.nn.functional.grid_sample``在特殊输入情况下报出的CUDA错误。([#33100](https://github.com/PaddlePaddle/Paddle/pull/33100))
- 修复``paddle.flatten``在静态图特殊输入情况下编译期计算维度错误的问题。([#35321](https://github.com/PaddlePaddle/Paddle/pull/35321))
- 修复``paddle.nn.conv2d/conv3d/conv2d\_transpose/conv3d\_transpose``计算输出shape时编译期检查报错的问题。([#35693](https://github.com/PaddlePaddle/Paddle/pull/35693))
- 修复``paddle.data.flowers``在多卡训练情况下容易出现数据读取错误的问题。([#33738](https://github.com/PaddlePaddle/Paddle/pull/33738))
- 修复pact量化se模块时loss为nan的问题。([#35392](https://github.com/PaddlePaddle/Paddle/pull/35392))
- 修复量化`flatten_contiguous_range`报错的问题。([35410](https://github.com/PaddlePaddle/Paddle/pull/35410))
- 修复动态图模式下pact量化的问题。([#35407](https://github.com/PaddlePaddle/Paddle/pull/35407))
- 修复channel-wise量化bert报错的问题。([#34948](https://github.com/PaddlePaddle/Paddle/pull/34948))
- 修复量化在参数全为0时的问题。([#34647](https://github.com/PaddlePaddle/Paddle/pull/34647))
- 修复channel-wise量化在channel数为1时的bug。([#33753](https://github.com/PaddlePaddle/Paddle/pull/33753))
- 修复动态图``@no_grad``线程不安全的问题。([#34649](https://github.com/PaddlePaddle/Paddle/pull/34649))
- 修复``paddle.grad``接口在部分场景下会hang住的bug。([#34023](https://github.com/PaddlePaddle/Paddle/pull/34023))
- 修复 ``paddle.masked_select``在静态图下形状推导的bug。([#33167](https://github.com/PaddlePaddle/Paddle/pull/33167))
- 修复``paddle.slice``在部分场景下不支持`numpy.ndarray`类型索引的问题，以及`axes`参数为`tuple`类型时出错的问题。([#35748](https://github.com/PaddlePaddle/Paddle/pull/35748), [#35267](https://github.com/PaddlePaddle/Paddle/pull/35267))
- 修复`set_value`反向梯度截断的问题。([#34304](https://github.com/PaddlePaddle/Paddle/pull/34304))
- 修复``paddle.regularizer.L1Decay`` 在非inplace计算下的gradient重复设置问题。 ([32710](https://github.com/PaddlePaddle/Paddle/pull/32710))
- 修复``adamw``参数分组时，学习率不生效问题。([#34468](https://github.com/PaddlePaddle/Paddle/pull/34468))
- 优化卷积类API中非法``dilate``输入检查。([#35894](https://github.com/PaddlePaddle/Paddle/pull/35894))
- 修复`paddle.io.DataLoader`迭代中途break报错问题。([#34501](https://github.com/PaddlePaddle/Paddle/pull/34501)) DataLoader内存泄漏问题。([#34140](https://github.com/PaddlePaddle/Paddle/pull/34140)) DataLoader误报warning信息。 ([#33712](https://github.com/PaddlePaddle/Paddle/pull/33712)) DataLoader子进程random state一致问题。([#33310](https://github.com/PaddlePaddle/Paddle/pull/33310))
- 修复IterableDataset中drop_last不生效问题。([#34801](https://github.com/PaddlePaddle/Paddle/pull/34801))
- 修复 ``paddle.optimizer.lr.LRScheduler`` 导致的 optimizer 状态恢复的问题。( [#33984](https://github.com/PaddlePaddle/Paddle/pull/33984))
- 修复``gather``算子，在使用`axis`进行infershape的bug。([#33413](https://github.com/PaddlePaddle/Paddle/pull/33413))
- 修复Executor中fetch_list类型为tuple时可能导致执行卡住的问题。([#35726](https://github.com/PaddlePaddle/Paddle/pull/35726))
- 修复``paddle.nn.GroupNorm``除零错误，并添加channel可以被group整除检查。([#35644](https://github.com/PaddlePaddle/Paddle/pull/35644))
- 修复tensor formatter中引用已释放内存的问题。([#35399](https://github.com/PaddlePddle/Paddle/pull/35399))
- 修复Adam优化器在``float64``精度下 `beta` 参数精度损失的问题。([#33381](https://github.com/PaddlePaddle/Paddle/pull/33381))
- 修复张量并行非切分参数初始化时未广播带来的精度对不齐问题。([#35326](https://github.com/PaddlePaddle/Paddle/pull/35326))
- 迁移``paddle.static.accuracy`` API中的`topk`算子到`topk_v2`算子。([#35494](https://github.com/PaddlePaddle/Paddle/pull/35494))
- 迁移``paddle.nn.dynamic_decode``中`expand`算子到`tile`算子，迁移`paddle.nn.BeamSearchDecoder`中`topk`算子到`topk_v2`算子。([#35656](https://github.com/PaddlePaddle/Paddle/pull/35656))
- 迁移``paddle.nn.functional.dice_loss``API中的`one_hot`算子到`one_hot_v2`算子。([#35734](https://github.com/PaddlePaddle/Paddle/pull/35734))
- 修复 ``paddle.summary`` 静态图模式下使用 bug。([#35303](https://github.com/PaddlePaddle/Paddle/pull/35303))
- 修复 ``paddle.Model.prepare`` 静态图模式下多卡启动的 bug。([#34311](https://github.com/PaddlePaddle/Paddle/pull/34311))
- 修复`paddle.nn.functional.cross_entropy` 给定`weight`，且指定`axis`为除-1外的其他合法维度时会报错的问题。([#36647](https://github.com/PaddlePaddle/Paddle/pull/36647))
- 修复`paddle.utils.dlpack.to_dlpack`无法编码多维 `Tensor` 的问题，修复其所生成的 DLPack 对象无法进行跨深度学习框架共享的问题。([#36177](https://github.com/PaddlePaddle/Paddle/pull/36177))
- 修复使用`paddle.distribution.Categorical`的`sample`方法报错的问题，具体原因是multinomial op的cuda kernel中数组访问越界，该bug会导致访问超出数组下标的值，引起报错。 ([#36511](https://github.com/PaddlePaddle/Paddle/pull/36511))
- 修复动态图`_BatchNormBase`基类中修改了 default_dtype，导致后续组网参数类型错误的问题，受影响的API有`paddle.nn.BatchNorm1D`，`paddle.nn.BatchNorm2D`，`paddle.nn.BatchNorm3D`，`paddle.nn.SyncBatchNorm`。具体原因是当 `get_default_dtype() == 'float16'` 时，通过 `set_default_dtype('float32')`修改默认参数数据类型，动态图组网的参数类型是通过 default_dtype 来创建的，因此当默认参数类型被修改后导致后续的组网参数类型错误。 ([#36376](https://github.com/PaddlePaddle/Paddle/pull/36376))
- 修复`paddle.nn.functional.grid_sample`因特殊输入导致的异常问题。([#36625](https://github.com/PaddlePaddle/Paddle/pull/36625))
- 修复 `paddle.fft.fft`, `paddle.fft.ifft`, `paddle.fft.rfft` , `paddle.fft.irfft`, `paddle.fft.hfft`, `paddle.fft.ihfft` 在输入 `axis=0` 情况下的计算错误问题。([#36537](https://github.com/PaddlePaddle/Paddle/pull/36537))
- 修复 `paddle.fft.fftshift`  和 `paddle.fft.ifftshift` 在静态图下出错的问题。([#36537](https://github.com/PaddlePaddle/Paddle/pull/36537))
- 修复 `paddle.fft.ifftshift` 计算结果不正确的问题。([#36835](https://github.com/PaddlePaddle/Paddle/pull/36835))
- 修复`paddle.nn.functional.pad`在`replicate`模式下的报错信息提示。([#36531](https://github.com/PaddlePaddle/Paddle/pull/36531))


#### IR(Intermediate Representation)

- 动态图转静态图
    - 修复了动转静后，在 ``paddle.no_grad`` 语义下显存异常增长的问题。([#35725](https://github.com/PaddlePaddle/Paddle/pull/35725))
    - 修复了对 ``paddle.no_grad`` 接口的错误识别和转换问题。([#34136](https://github.com/PaddlePaddle/Paddle/pull/34136)) 
    - 修复了部分场景下模型中间设置 stop_gradient=True 时，动转静训练报错的问题。([#36353](https://github.com/PaddlePaddle/Paddle/pull/36353))
    - 修复了在控制流 if 的部分场景转换时，对返回结果检查会报错的问题。([#36830](https://github.com/PaddlePaddle/Paddle/pull/36830))
    - 修复了在 ifelse 分支返回不等长结果时，动转静会额外对齐返回长度导致返回类型意外改变的问题。([#36565](https://github.com/PaddlePaddle/Paddle/pull/36565))
    - 修复使用 jit.save/load 接口加载模型后，在 train 模式和 no_grad 上下文中，显存会一直增长的问题。([#36463](https://github.com/PaddlePaddle/Paddle/pull/36463))


#### 分布式训练

- 分布式训练基础功能
    - 修复图引擎潜在的栈溢出问题。 ([#33055](https://github.com/PaddlePaddle/Paddle/pull/33055)) 
    - 修复分布式训练可能出现的死锁问题。 ([#34461](https://github.com/PaddlePaddle/Paddle/pull/34461))
    - 修复张量并行在 transformer 类模型的多头注意力计算中切分不正确的问题，优化张量并行在混合精度计算时的速度。 ([#33015](https://github.com/PaddlePaddle/Paddle/pull/33015)) 
    - 修复模型并行下使用 `paddle.nn.ClipGradientByGlobalNorm` 时，非 distributed 的 vars 的 norm 被多次计算的问题。([#35713](https://github.com/PaddlePaddle/Paddle/pull/35713))
    - 修复模型并行`paddle.distributed.split` Parallel Linear 行切分bias加法位置出错的问题。([#35186](https://github.com/PaddlePaddle/Paddle/pull/35186))
    - 修复流水线并行初始化通信组可能 hang 的问题。 ([#33476](https://github.com/PaddlePaddle/Paddle/pull/33476))
    - 修复流水线并行中 `Tensor` 显存在实际使用完成前被释放的问题。 ([#33996](https://github.com/PaddlePaddle/Paddle/pull/33996))
    - 修复流水线并行时反向梯度累加 `op_device`为空的问题。([#33875](https://github.com/PaddlePaddle/Paddle/pull/33875))
    - 修复流水线并行运行 `sub-block` 报错的问题。([#32727](https://github.com/PaddlePaddle/Paddle/pull/32727))
    - 修复流水线并行时反向梯度累加 `op_device`为空的问题。([#33875](https://github.com/PaddlePaddle/Paddle/pull/33875))
    - 修复 Sharding 并行通信初始化时偶尔 hang 住的问题。 ([#33327](https://github.com/PaddlePaddle/Paddle/pull/33327))
    - 修复 `paddle.distributed.barrier` 同步流错误。 ([#33476](https://github.com/PaddlePaddle/Paddle/pull/33476))
    - 修复 `paddle.distributed.alltoall` 通信组设置错误的问题。([#32890](https://github.com/PaddlePaddle/Paddle/pull/3492890))
    - 修复静态图张量并行参数初始换广播错误导致的精度对不齐问题。([35326](https://github.com/PaddlePaddle/Paddle/pull/35326))
    - 修复动态图数据并行不支持 `recompute` 等继承 `PyLayer` 类实现的自定义算子的问题。([#35401](https://github.com/PaddlePaddle/Paddle/pull/35401))
    - 修复混合并行下流水线并行 + 数据并行 hang 住的问题。([#34142](https://github.com/PaddlePaddle/Paddle/pull/34142))
    - 修复开启 AMP 时，`fleet.get_loss_scaling` 失败的问题。([#33935](https://github.com/PaddlePaddle/Paddle/pull/33935))
    - 修复 `fleet` 多机未 wait server ready 的问题。([#32889](https://github.com/PaddlePaddle/Paddle/pull/32889))
    - 修复分布式预测 `infer_from_dataset` 仍旧更新参数梯度的问题。([#35698](https://github.com/PaddlePaddle/Paddle/pull/35698))
    - 修复 `data_feed` 中 dense 特征 LOD 属性设置错误的问题。([#35000](https://github.com/PaddlePaddle/Paddle/pull/35000))
    - 修复静态图使用 `gradientmerge`时 `gradient_merge_cond` 变量的 save 问题。([#35578](https://github.com/PaddlePaddle/Paddle/pull/35578))
    - 修复 `paddle.hub`下载文件名字和 `nt_merge_cond` 变量的 save 问题。([#35578](https://github.com/PaddlePaddle/Paddle/pull/35578))
    - 修复 `fleet` 开启 `dump_slot` 时报错不明显的问题。 ([#34173](https://github.com/PaddlePaddle/Paddle/pull/34173))
    - 修复混合并行训练在海光 DCU 芯片上的 RCCL 的问题。([#32808](https://github.com/PaddlePaddle/Paddle/pull/32808))
    - 修复 GPU 参数服务器退出报错问题。([#33724](https://github.com/PaddlePaddle/Paddle/pull/33724))
    - 修复 hdfs 工具upload/download功能不可用问题。([#33903](https://github.com/PaddlePaddle/Paddle/pull/33903))
    - 修复 GPU 参数服务器训练过程中由于样本不能整除worker数而卡住的问题。([#32640](https://github.com/PaddlePaddle/Paddle/pull/32640))
    - 修复 GPU 参数服务器使用非0卡训练报错问题。([#33078](https://github.com/PaddlePaddle/Paddle/pull/33078))
    - 修复 GPU 参数服务器 delta score，scale show问题。([#33492](https://github.com/PaddlePaddle/Paddle/pull/33078), [#33492](https://github.com/PaddlePaddle/Paddle/pull/33492))
    - 修复 GPU 参数服务器训练结束后未 merge dense，g2sum 计算有误，data norm 添加了optimize op 等问题。 ([#35029](https://github.com/PaddlePaddle/Paddle/pull/35029))
    - 修复使用 fuse all reduce ops 开关时，如果梯度出现 empty 时会报错的问题。([#36231](https://github.com/PaddlePaddle/Paddle/pull/36231))
    - 修复 dist_transformer 文件出现未定义的变量问题。([#36211](https://github.com/PaddlePaddle/Paddle/pull/36211))
	
	
	
- 动态图混合并行
	- 修复流水线并行计算错误的问题。([#35556](https://github.com/PaddlePaddle/Paddle/pull/35556))
	- 修复张量并行下，c_spilt 的方向计算的问题。([#33207](https://github.com/PaddlePaddle/Paddle/pull/33207))
	- 修复张量并行下，精度无法对齐的问题。([#32897](https://github.com/PaddlePaddle/Paddle/pull/32897))
	- 修复new_group() 创建通信组创建时，出现随机hang的情况。([#33141](https://github.com/PaddlePaddle/Paddle/pull/33141))
	- 修复数据并行下 reducer 遍历反向图的问题。( [#32715](https://github.com/PaddlePaddle/Paddle/pull/32715))
	- 修复数据并行下参数同步的属性缺失的问题。 ([#33955](https://github.com/PaddlePaddle/Paddle/pull/33955))

- 静态图混合并行
    - 解决 TensorParallel 在 Multi-Head Attention 网络中的切分错误问题，优化 TensorParallel 与混合精度共同使用时的训练速度。([#32897](https://github.com/PaddlePaddle/Paddle/pull/32897))
	
#### 其他
- 自定义OP
    - 修复 ``paddle::Tensor`` 的 ``cast`` 方法在 GPU 下不生效的问题。([#34884](https://github.com/PaddlePaddle/Paddle/pull/34884))
    - 修复自定义算子不能同时加载多个模块的问题。([#34505](https://github.com/PaddlePaddle/Paddle/pull/34505))
    - 修复联合编译 .cc 和 .cu 文件时，``PADDLE_WITH_CUDA`` 宏未生效的问题。([#35448](https://github.com/PaddlePaddle/Paddle/pull/35448))
- 去除对 ``logging`` 库全局设置的修改。 ([#32673](https://github.com/PaddlePaddle/Paddle/pull/32673))
- 新增 ``GlooParallelContext``，适配 `Reducer` 模块逻辑，为 `DataParallel` 后续支持CPU并行提供底层通信组件支持。 ([#35154](https://github.com/PaddlePaddle/Paddle/pull/35154))
- 迁移 `paddle.metric.accuracy` 中的 `top_k` op 为 `top_k_v2` op。 ([#35789](https://github.com/PaddlePaddle/Paddle/pull/35789))
- 修复 `MKLDNN` 下运行找不到默认 `attr` 的问题。([#34567](https://github.com/PaddlePaddle/Paddle/pull/34567))
- 修复 `optimizer` 中没有给 `clear_float_status` OP添加 `device_key` 的问题。([#34431](https://github.com/PaddlePaddle/Paddle/pull/34431))



## 4. 部署方向（Paddle Inference）
### （1）新增功能

#### 后端能力增强
- 新增 TensorRT 子图模式下动态 shape 自动配置功能
  增加TensorRT离线tune动态shape设置方式，对于模型被切分成多个TensorRT子图的场景，提升易用性[#34806](https://github.com/PaddlePaddle/Paddle/pull/34806) [#35771](https://github.com/PaddlePaddle/Paddle/pull/35771)，使用示例可参考[demo](https://github.com/PaddlePaddle/Paddle-Inference-Demo/tree/master/c%2B%2B/paddle-trt/tuned_dynamic_shape)。

    - 易用性优化的基本思想是：使用Paddle原生运行的方式针对用户输入的批量数据，统计计算图中所有临时tensor的shape范围，并将统计到的shape范围设置到TensorRT子图的输入，从而避免了用户去手动计算内部子图输入tensor的shape范围，提升易用性。
    - 离线tuned动态shape使用的基本流程：用户代码完成后，通过配置config，启用shape范围收集能力c++接口`config.CollectShapeRangeInfo("shape_range.pbtxt")`或python接口`config.collect_shape_range_info('shape_range.pbtxt')`，将获得的shape范围以prototxt的格式存储在本地，修改config配置，关闭shape收集，开启tensorrt和动态shape能力，c++接口`config.EnableTunedTensorRtDynamicShape("shape_range.pbtxt", true)`或python接口`config.enable_tuned_tensorrt_dynamic_shape('shape_range.pbtxt', True)`即可直接运行。  


- 新增对昇腾(Ascend)系列硬件的原生支持
    - 子图通过支持Paddle-Lite NNAdapter接入ascend310硬件预测 [#35226](https://github.com/PaddlePaddle/Paddle/pull/35226)， 示例可参考[demo](https://github.com/PaddlePaddle/Paddle-Inference-Demo/tree/master/c%2B%2B/ascend310_lite_subgraph/image_classification_demo)。
    - 新增晟腾910 推理支持 [#34101](https://github.com/PaddlePaddle/Paddle/pull/34101)

- 新增pool3d算子支持TensorRT的功能。([#36545](https://github.com/PaddlePaddle/Paddle/pull/36545))

### （2）功能优化

#### 框架及API更新

- 量化支持 
    - 动态图量化推理 pass 的重构，支持非模拟量化的 OP和模拟量化的 OP。([#35907](https://github.com/PaddlePaddle/Paddle/pull/35907))
  - 增加 int8 的模拟量化OP matmul（权重乘以 tensor的情况）。([#34359](https://github.com/PaddlePaddle/Paddle/pull/34359))
  - 修复MobileNetV3模型在量化训练过程中因量化参数为0导致的Loss出NAN问题。([#36763](https://github.com/PaddlePaddle/Paddle/pull/36763))


- API 增强
    - 基于新版CAPI重构GO API，[#33113](https://github.com/PaddlePaddle/Paddle/pull/33113)，使用示例可参考[demo](https://github.com/PaddlePaddle/Paddle-Inference-Demo/tree/master/go/resnet50)。
    - 预测python api `copy_from_cpu` 和 `copy_to_cpu` 接口支持float16数据类型 。([#34676](https://github.com/PaddlePaddle/Paddle/pull/34676))
    - 增加 `config.Summary()` 接口，打印config配置信息。([#34122](https://github.com/PaddlePaddle/Paddle/pull/34122))
    - 预测库中 `version.txt` 记录trt版本信息补全，如v7.2.3.4 而不是v7。( [#33690](https://github.com/PaddlePaddle/Paddle/pull/33690))

- 库体积压缩
    - linux 下对预测库进行strip裁剪库体积，体积压缩30M。([#34895](https://github.com/PaddlePaddle/Paddle/pull/34895))

- 其他更新
    - 新增捕获运行异常报错并将其转换为相应错误状态的辅助工具。([#35624](https://github.com/PaddlePaddle/Paddle/pull/35624))
    - 新增相关基础数据结构，增强飞桨算子定义的精确性。([#33098](https://github.com/PaddlePaddle/Paddle/pull/33098))

#### 后端能力增强

- CPU 相关更新
    - 升级oneDNN版本为2.3.2。( [#35040](https://github.com/PaddlePaddle/Paddle/pull/35040))
    - 新增 quant-aware LSTM oneDNN INT8 模型支持。([#35382](https://github.com/PaddlePaddle/Paddle/pull/35382))
    - 新增 post-training LSTM oneDNN INT8 模型支持。([#35334](https://github.com/PaddlePaddle/Paddle/pull/35334), [#33295](https://github.com/PaddlePaddle/Paddle/pull/33295))
    - 新增 fusion_gru 和 multi_gru 融合和 post-training INT8的支持。([#33749](https://github.com/PaddlePaddle/Paddle/pull/33749))
    - 优化oneDNN 的 cache机制。([#35664](https://github.com/PaddlePaddle/Paddle/pull/35664),  [#35331](https://github.com/PaddlePaddle/Paddle/pull/35331), [#35132](https://github.com/PaddlePaddle/Paddle/pull/35132), [#35030](https://github.com/PaddlePaddle/Paddle/pull/35030), [#35002](https://github.com/PaddlePaddle/Paddle/pull/35002), [#34830](https://github.com/PaddlePaddle/Paddle/pull/34830), [#33515](https://github.com/PaddlePaddle/Paddle/pull/33515), [#33048](https://github.com/PaddlePaddle/Paddle/pull/33048), [#32922](https://github.com/PaddlePaddle/Paddle/pull/32922), [#32499](https://github.com/PaddlePaddle/Paddle/pull/32499))
    - 通过新增多个 op (如clip, scale等) 的oneDNN kernel 实现,  ch_ppocr_mobile_v1.1_det_infer、DPN68, fastscnn, hrnet、HRNet_W18_C、 icnet、Res2Net50_26w_4s、 ssdlite_mobilenet_v3_large 等模型打开oneDNN 比关闭 oneDNN 在 Intel(R) Xeon(R) Gold 6271C CPU @ 2.60GHz 单核性能提升 47.8%。([#35601](https://github.com/PaddlePaddle/Paddle/pull/35601), [#32975](https://github.com/PaddlePaddle/Paddle/pull/32975))
    - 优化了oneDNN LSTM INT8 模型，在Intel(R) Xeon(R) Gold 6271C CPU @ 2.60GHz 单核上，INT8 LSTM 模型为 FP32 LSTM 模型性能的 1.59 倍。([#35382](https://github.com/PaddlePaddle/Paddle/pull/35382), [#35334](https://github.com/PaddlePaddle/Paddle/pull/35334), [#34820](https://github.com/PaddlePaddle/Paddle/pull/34820), [#34137](https://github.com/PaddlePaddle/Paddle/pull/34137))


- GPU 及 TensorRT 子图引擎相关更新

    - 增加TensorRT 8.0的支持，在将来的某个版本我们会放弃对TensorRT 6.x的支持。([#34403](https://github.com/PaddlePaddle/Paddle/pull/34403), [#34294](https://github.com/PaddlePaddle/Paddle/pull/34294), [#34157](https://github.com/PaddlePaddle/Paddle/pull/34157), [#33777](https://github.com/PaddlePaddle/Paddle/pull/33777), [#33680](https://github.com/PaddlePaddle/Paddle/pull/33680), [#33662](https://github.com/PaddlePaddle/Paddle/pull/33662), [#33654](https://github.com/PaddlePaddle/Paddle/pull/33654))
  - 增加TensorRT `layer_norm` plugin对动态shape的支持。([#33448](https://github.com/PaddlePaddle/Paddle/pull/33448))
  - 增加TensorRT `hard_swish` plugin对动态shape的支持。([#35214](https://github.com/PaddlePaddle/Paddle/pull/35214))
  - 增加TensoRT `reduce_sum` 和 `gather_nd` 的支持。([#33324](https://github.com/PaddlePaddle/Paddle/pull/33324))
  - 增加TensorRT `qkv_context` plugin 对int8的支持([#34917](https://github.com/PaddlePaddle/Paddle/pull/34917), [#35504](https://github.com/PaddlePaddle/Paddle/pull/35504))
  - 增加TensorRT conv3d的支持。([#35507](https://github.com/PaddlePaddle/Paddle/pull/35507))
  - 增加对 `multihead_matmul` 融合算子的输入进行广播的支持。([#35780](https://github.com/PaddlePaddle/Paddle/pull/35780))
  - Inference 支持 TensorRT8 稀疏推理，[测试环境](https://github.com/PaddlePaddle/Paddle-Inference-Demo/tree/master/c%2B%2B/sparsity)下，ERNIE 模型变长输入在不同的 batch_size 下性能提升10%-30%，ResNeXt101_32x4d模型在不同的batch_size下性能提升10%。([#36659](https://github.com/PaddlePaddle/Paddle/pull/36659))

- Nvidia Jetson 原生支持能力增强
    - 新增 Op 支持，针对Jetson Nano/TX2这两款算力较低的设备，我们做了针对性的优化，目前新增了 `pool2d`, `pool_max`, `conv3d_transpose` 等 17个OP的支持。([#35378](https://github.com/PaddlePaddle/Paddle/pull/35378))
    - 针对Jetson Nano，新增模型：DPN68, EfficientNetB0, ttfnet, fcn_hrnetw18, hardnet。([#35378](https://github.com/PaddlePaddle/Paddle/pull/35378))
    - 针对Jetson TX2，新增模型：deeplabv3p_resnet50, deeplabv3_resnet50, fcn_hrnetw18, hardnet, pspnet, ttfnet, unet。([#35378](https://github.com/PaddlePaddle/Paddle/pull/35378))

- 昆仑XPU接口功能扩展 
  - 新增 `set_xpu_device_id` 接口，支持设置推理时的昆仑芯片的设备号([#35572](https://github.com/PaddlePaddle/Paddle/pull/35572))

- Inference python `copy_from_cpu`接口加入输入类型检查，错误类型输入下提前报错。([#36552](https://github.com/PaddlePaddle/Paddle/pull/36552))

### （3）问题修复

#### 框架及API修复

- 算子修复
    - 修复split op当axis输入小于0时，转换TensorRT时会发生地址访问错误的情况，同时将axis等于0时静动态shape均不支持的情况进行过滤。([#35127](https://github.com/PaddlePaddle/Paddle/pull/35127))
    - 修复transpose静态shape在axis为`[0, 1]`时错误的情况。([#35138](https://github.com/PaddlePaddle/Paddle/pull/35138))
    - 修复 gather op与原生 paddle op的功能对齐，并完善 op teller 过滤的条件。([#35784](https://github.com/PaddlePaddle/Paddle/pull/35784))
    - 修复fc op 的 int8 分支。([#34787](https://github.com/PaddlePaddle/Paddle/pull/34787), [#32671](https://github.com/PaddlePaddle/Paddle/pull/32671))
    - 修复reshape 的 op teller 过滤条件。([#34787](https://github.com/PaddlePaddle/Paddle/pull/34787), [#34583](https://github.com/PaddlePaddle/Paddle/pull/34583))
    - 修复recurrent op多线程推理效率差问题。([#36053](https://github.com/PaddlePaddle/Paddle/pull/36053))
    - 修复gather和scatter op中int值溢出的问题。([#35544](https://github.com/PaddlePaddle/Paddle/pull/35544))
    - 修复 ctc op 除零错误。 ([#34724](https://github.com/PaddlePaddle/Paddle/pull/34724))
    - 修复模型输入包含bool类型时，插入scale op导致的崩溃。([#35176](http://github.com/PaddlePaddle/Paddle/pull/35176))
    - 修复复数scaler 和Tensor 运算失败的问题。([#33699](https://github.com/PaddlePaddle/Paddle/pull/33699))

- 框架功能修复
    - 修复部分ernie模型批处理数据时显存访问越界的问题。([#35077](https://github.com/PaddlePaddle/Paddle/pull/35077))
    - 修复ernie模型FP16精度运行时可能出现的精度问题。([#34771](https://github.com/PaddlePaddle/Paddle/pull/34711))
    - 修复ernie变长情况下，输入的顺序不一致导致输出不对的问题。([#33575](https://github.com/PaddlePaddle/Paddle/pull/33575))
    - 修复多流状态下分配器功能异常的问题。([#32932](https://github.com/PaddlePaddle/Paddle/pull/33575))

- 修复 ERNIE 模型在 TRT8 下可能出现的崩溃问题。([#36769](https://github.com/PaddlePaddle/Paddle/pull/36769))
- 修复使用 Pool, Slice 时可能出现的崩溃及精度问题。([#36666](https://github.com/PaddlePaddle/Paddle/pull/36666))
- 修复 yolo_box op因为计算公式错误导致的精度问题。([#36365](https://github.com/PaddlePaddle/Paddle/pull/36365))
- 修复量化后的 matmul_v2 在TRT下无法正常推理的问题。([#36821](https://github.com/PaddlePaddle/Paddle/pull/36821))
- 修复了量化 matmul_v2 时错误地添加量化op的问题。([#36820](https://github.com/PaddlePaddle/Paddle/pull/36820))
- 修复算子 batch_norm 和 elementwise_add 在3D应用场景下开启 TRT 报错的问题。([#36446](https://github.com/PaddlePaddle/Paddle/pull/36446))
- 修复高层 linear api保存得到的预测模型无法被 Pass 融合优化的问题。([#36500](https://github.com/PaddlePaddle/Paddle/pull/36500))
- 修改 MatmulV2ToMul 的 Pass，重新限定 (matmul_v2 to mul) 映射的 Pass，增加 MatmulV2ToMatmul 的 Pass，限定 (matmul_v2 to matmul) 映射的 Pass条件(不支持广播)，修改 (matmul, mul) 的 op_teller 映射条件。([#36652](https://github.com/PaddlePaddle/Paddle/pull/36652))


#### 后端能力修复

- TensorRT 子图引擎修复
    - 修复TensorRT动态shape时slice plugin的ends参数越界报错问题。([#35357](https://github.com/PaddlePaddle/Paddle/pull/35357))
    - 修复reduce op转换TensorRT的reduce_all = 1时候不支持keepdim=false的情况。([#35145](https://github.com/PaddlePaddle/Paddle/pull/35145))
    - 修复slice op转换TensorRT的decrease_axis参数问题。([#35100](https://github.com/PaddlePaddle/Paddle/pull/35100))
    - 修复nearest_interp op转换TensorRT动态shape下scale为负数不支持的情况。修正scale比outh和outw有更高优先级。([#35405](https://github.com/PaddlePaddle/Paddle/pull/35405))
    - 修复pad op的paddings参数和tensorrt不一样的问题。([#35371](https://github.com/PaddlePaddle/Paddle/pull/35371))
    - 添加conv2d op转换TensorRT的4维padding支持，过滤conv2d op转换TensorRT时padding_algorithm 为 SAME 和 VALID 的情况。([#35627](https://github.com/PaddlePaddle/Paddle/pull/35627))
    - 添加pool2d op转换TensorRT时对padding_algorithm 为 SAME 的处理，过滤 exclusive mode下 ksize 小于等于 padings 的情况。([#35923](https://github.com/PaddlePaddle/Paddle/pull/35923))
    - 修复clip op转换TensorRT时不支持 Min和Max 输入的情况。([#35694](https://github.com/PaddlePaddle/Paddle/pull/35694))
    - 修复gelu op转换TensorRT时不支持 approximate 属性的情况。([#35529](https://github.com/PaddlePaddle/Paddle/pull/35529))
    - 修复affine_channel转换TensorRT时不支持2维输入的情况。([#35496](https://github.com/PaddlePaddle/Paddle/pull/35496))
    - 修复TensorRT子图匹配不稳定的问题。([#35147](https://github.com/PaddlePaddle/Paddle/pull/35147))
    - 修复预测引擎析构后，TensorRT engine没有释放的问题。([#35842](https://github.com/PaddlePaddle/Paddle/pull/35842), [#35938](https://github.com/PaddlePaddle/Paddle/pull/35938))
    - paddle-trt static模式下，如果reshape的shape属性 batch维度为-1，修复paddle算子错误转换为trt的问题。([#34007](https://github.com/PaddlePaddle/Paddle/pull/34007))
    - 修复roi_align 转换TensorRT不支持RoisNum属性的情况，同时修复在动态shape时aligned 为True、Sampling_ratio = -1计算错误的情况。([#35549](https://github.com/PaddlePaddle/Paddle/pull/35549))
    - 修复concat 转换TensorRT不支持AxisTensor属性的情况。([#35545](https://github.com/PaddlePaddle/Paddle/pull/35545))
    - 修复scale 转换TensorRT不支持ScaleTensor属性以及静态shape 不支持1维输入的情况。([#35225](https://github.com/PaddlePaddle/Paddle/pull/35225))
    - 修复batchnorm 转换TensorRT不支持MomentumTensor属性的情况。([#35527](https://github.com/PaddlePaddle/Paddle/pull/35527))
    - 修复reshape 转换TensorRT不支持ShapeTensor 、Shape属性以及静态shape 不支持1维输入的情况。([#35166](https://github.com/PaddlePaddle/Paddle/pull/35166))
    - 增加 TensorRT tile 算子支持。([#34388](https://github.com/PaddlePaddle/Paddle/pull/34388))
    - 增加 TensorRT reduce mean 算子支持。([#34204](https://github.com/PaddlePaddle/Paddle/pull/34204))
    - 修复使用gather op时可能出现的崩溃问题。([#33999](https://github.com/PaddlePaddle/Paddle/pull/33999))
    - 修复 TensorRT int8 的一个错误使用 debug 的 flag（会只运行 int8的 kernel，导致性能下降）。([#34704](https://github.com/PaddlePaddle/Paddle/pull/34704))
    - 修复gather_nd op在2维输入调用TensorRT时计算错误问题。([#35464](https://github.com/PaddlePaddle/Paddle/pull/35464))
    - 修复hard_sigmoid op在2维输入调用TensorRT时计算错误问题。([#35908](https://github.com/PaddlePaddle/Paddle/pull/35908))
    - 修复prelu op在2维输入调用TensorRT时计算错误问题。([#35512](https://github.com/PaddlePaddle/Paddle/pull/35512))
    - 修复windows下 TensorRT 推理时，有用右斜杠作为路径分隔符导致的崩溃问题。([#33853](http://github.com/PaddlePaddle/Paddle/pull/33853))


#### 其他修复

- 修复裁剪反向算子脚本遇到中文字符注释出错的问题。([#33937](https://github.com/PaddlePaddle/Paddle/pull/33937), [#33919](https://github.com/PaddlePaddle/Paddle/pull/33919))
- 修复编译时单测模型下载不全导致单测推理时的错误，增加测试模型下载的 MD5下载验证。([#33264](https://github.com/PaddlePaddle/Paddle/pull/33264), [#33217](https://github.com/PaddlePaddle/Paddle/pull/33217))
- 修复 blazeface model 中mkldnn elementwise op 不支持 broadcast 问题。([#33549](https://github.com/PaddlePaddle/Paddle/pull/33549))
- 修复 swin_transformer mkldnn 推理报错问题。([#35740](https://github.com/PaddlePaddle/Paddle/pull/35740))
- 修复 paddlex.deploy.Predictor oneDNN多线程执行 unet 报错问题。([#35231](https://github.com/PaddlePaddle/Paddle/pull/35231))
- 修复 oneDNN setCacheCapacity无法限制内存问题。([#33571](https://github.com/PaddlePaddle/Paddle/pull/33571))




## 环境适配

### 编译安装
- Windows 全新支持 `Ninja编译构建方式`，编译速度、易用性、稳定性都较VS IDE方式有很好提升，Windows用户可`pip install ninja`，进行本地源码编译Paddle。([#31161](https://github.com/PaddlePaddle/Paddle/pull/31161), [#31449](https://github.com/PaddlePaddle/Paddle/pull/31449), [#32987](https://github.com/PaddlePaddle/Paddle/pull/32987), [#33140](https://github.com/PaddlePaddle/Paddle/pull/33140), [#33155](https://github.com/PaddlePaddle/Paddle/pull/33155))
- 发版镜像中只保留python3.7，删除了python3.5、python3.6、python3.8、python3.9及相应python版本的paddle包，缩小镜像大小。镜像大小缩小30%~50%。([#32688](https://github.com/PaddlePaddle/Paddle/pull/32688))
- TensorRT库为推理时使用，发版镜像中仅paddle训练基础功能，不需要支持TensorRT。删除了发版镜像中的TensorRT库，避免用户错误使用该镜像。([#34266](https://github.com/PaddlePaddle/Paddle/pull/34266))

### 新硬件适配

- 海光DCU芯片训练和推理支持，支持模型数量达9个分类70个模型。
    - 海光DCU新增 PaddleDetection 模型支持5个。
    - 海光DCU新增 PaddleGAN 模型支持6个。
    - 海光DCU新增 PaddleSeg 模型支持13个。
    - 海光DCU新增 PaddleNLP 模型支持3个。
    - 海光DCU新增 PaddleOCR 模型支持4个。
    - 海光DCU新增 PaddleVideo 模型支持3个。
- 昆仑芯第2代芯片(XPU-2)训练支持，支持ResNet50、SSD、Bert、Transformer等多个模型 ，支持静态图+动态图训练，支持混合精度训练。

## Thanks to our Contributors

This release contains contributions from:

0x45f, 123malin, Adam Osewski, Aganlengzi, Aurelius84, Baibaifan, Bo Liu, CheQiXiao, Chen Long, Chen Weihang, CtfGo, Double\_V, Ethanzjp, Fan Zhang, Feiyu Chan, Feng Xing, From00, GT-Zhang, Guanghua Yu, Guoxia Wang, Haipeng Wang, Hao Lin, Haohongxiang, Hui Zhang, Huihuang Zheng, HydrogenSulfate, IMMORTAL, JYChen, JZ-LIANG, Jacek Czaja, Jack Zhou, Jackwaterveg, Jeng Bai-Cheng, Jiangxinz, Jiaqi Liu, Jiawei Wang, JingZhuangzhuang, June Weng, Kaipeng Deng, Kqnonrime, LJQ❤️, Leo Chen, Li Min, LielinJiang, Lijunhui, Linjie Chen, Liu-xiandong, LiuWei, Ming-Xu Huang, MissPenguin, PaddlePM, Pei Yang, Peihan, Qi Li, QingshuChen, Ren Wei (任卫), Roc, Shang Zhizhou, ShenLiang, Shibo Tao, Siming Dai, Sing\_chan, TCChenLong, TTerror, TeslaZhao, Thomas Young, Thunderbrook, Tongxin Bai, WJJ1995, WangXi, Wangzheee, Wei Shengyu, WeiXin, Weilong Wu, Wenyu, Wilber, XGZhang, XYZ, XYZ916829, XiangGao, Xiaoxu Chen, YUNSHEN XIE, Yanxing Shi, Yiqun Liu, YuanRisheng, Yuang Liu, Yulong Ao, Zeng Jinle, Zhang Ting, Zhang Zheng, Zhanlue Yang, Zhen Wang, Zhong Hui, Zhou Wei, andreazanetti, andyjpaddle, arlesniak, baoachun, cc, ceci3, chajchaj, chenenquan, chenjian, chentianyu03, crystal, cuicheng01, danleifeng, denglin-github, duanboqiang, dyning, feng626, feng_shuai, furnace, gongweibao, heliqi, hlygit66666, hong, hong19860320, houj04, huangjun12, huangxu96, huzhiqiang, iducn, jakpiase, jiangcheng, joanna.wozna.intel, jzhang533, kuizhiqing, levi131, lidanqing, lilong12, limingshu, littletomatodonkey, liu zhengxi, liutiexing, liuyuhui, liym27, lyuwenyu, lzzyzlbb, niuliling123, pangyoki, parap1uie-s, ronnywang, root, seemingwang, shangliang Xu, shiyutang, smallv0221, sunli, sunzhongkai588, taixiurong, tangwei12, tianshuo78520a, veyron95, wangguanqun, wangguanzhong, wanghuancoder, wangna11BD, wangxinxin08, wangzhen38, wangzhuang01, wawltor, wenbin, whs, will-jl944, wuhuachaocoding, wuhuanzhou, xiaoting, xiaoxiaohehe001, xiayanming, xiegegege, xiemoyuan, xiongkun, yaoxuefeng, yeliang2258, yingyibiao, zhangbo9674, zhangchunle, zhangkaihuo, zhaoyingli, zhiboniu, zhoujun, zhouzj, zhulei, zhupengyang, zlsh80826, zmx, zyfncg, 李季, 津, 王明冬, 石晓伟

