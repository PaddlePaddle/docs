# 组网、训练、评估常见问题

##### 问题：`stop_gradient=True`的影响范围？

+ 答复：如果静态图中某一层使用`stop_gradient=True`，那么这一层之前的层都会自动 `stop_gradient=True`，梯度不再回传。

----------

##### 问题：请问`paddle.matmul`和`paddle.multiply`有什么区别？

+ 答复：`matmul`支持的两个 tensor 的矩阵乘操作。`muliply`是支持两个 tensor 进行逐元素相乘。

----------

##### 问题：请问`paddle.gather`和`torch.gather`有什么区别？

+ 答复：[`paddle.gather`](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/gather_cn.html#gather)和`torch.gather`的函数签名分别为：

```python
paddle.gather(x, index, axis=None, name=None)
torch.gather(input, dim, index, *, sparse_grad=False, out=None)
```

其中，`paddle.gather`的参数`x`，`index`，`axis`分别与`torch.gather`的参数`input`，`index`，`dim`意义相同。

两者在输入形状、输出形状、计算公式等方面都有区别，具体如下：

- `paddle.gather`

  - 输入形状：`x`可以是任意的`N`维 Tensor。但`index`必须是形状为`[M]`的一维 Tensor，或形状为`[M, 1]`的二维 Tensor。

  - 输出形状：输出 Tensor `out`的形状`shape_out`和`x`的形状`shape_x`的关系为：`shape_out[i] = shape_x[i] if i != axis else M`。

  - 计算公式：`out[i_1][i_2]...[i_axis]...[i_N] = x[i_1][i_2]...[index[i_axis]]...[i_N]` 。

  - 举例说明：假设`x`的形状为`[N1, N2, N3]`，`index`的形状为`[M]`，`axis`的值为 1，那么输出`out`的形状为`[N1, M, N3]`，且`out[i_1][i_2][i_3] = x[i_1][index[i_2]][i_3]`。

- `torch.gather`

  - 输入形状：`input`可以是任意的`N`维 Tensor，且`index.rank`必须等于`input.rank`。

  - 输出形状：输出 Tensor `out`的形状与`index`相同。

  - 计算公式：`out[i_1][i_2]...[i_dim]...[i_N] = input[i_1][i_2]...[index[i_1][i_2]...[i_N]]...[i_N]`。

  - 举例说明：假设`x`的形状为`[N1, N2, N3]`，`index`的形状为`[M1, M2, M3]`，`dim`的值为 1，那么输出`out`的形状为`[M1, M2, M3]`，且`out[i_1][i_2][i_3] = input[i_1][index[i_1][i_2][i_3]][i_3]`。

- 异同比较

  - 只有当`x.rank == 1`且`index.rank == 1`时，`paddle.gather`和`torch.gather`功能相同。其余情况两者无法直接互换使用。

  - `paddle.gather`不支持`torch.gather`的`sparse_grad`参数。

----------

##### 问题：在模型组网时，inplace 参数的设置会影响梯度回传吗？经过不带参数的 op 之后，梯度是否会保留下来？

+ 答复：inplace 参数不会影响梯度回传。只要用户没有手动设置`stop_gradient=True`，梯度都会保留下来。

----------

##### 问题：如何不训练某层的权重？

+ 答复：在`ParamAttr`里设置`learning_rate=0`或`trainable`设置为`False`。具体请[参考文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/ParamAttr_cn.html#paramattr)

----------

##### 问题：使用 CPU 进行模型训练，如何利用多处理器进行加速？

+ 答复：在 2.0 版本动态图模式下，CPU 训练加速可以从以下两点进行配置：

1. 使用多进程 DataLoader 加速数据读取：训练数据较多时，数据处理往往会成为训练速度的瓶颈，paddle 提供了异步数据读取接口 DataLoader，可以使用多进程进行数据加载，充分利用多处理的优势，具体使用方法及示例请参考 API 文档：[paddle.io.DataLoader](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/io/DataLoader_cn.html#dataloader)。

2. 推荐使用支持[MKL（英特尔数学核心函数库）](https://software.intel.com/content/www/us/en/develop/tools/oneapi/components/onemkl.html)的 paddle 安装包，MKL 相比 Openblas 等通用计算库在计算速度上有显著的优势，能够提升您的训练效率。

----------

##### 问题：使用 NVIDIA 多卡运行 Paddle 时报错 Nccl error，如何解决？

+ 答复：这个错误大概率是环境配置不正确导致的，建议您使用 NVIDIA 官方提供的方法参考检测自己的环境是否配置正确。具体地，可以使用[ NCCL Tests ](https://github.com/NVIDIA/nccl-tests) 检测您的环境；如果检测不通过，请登录[ NCCL 官网 ](https://developer.nvidia.com/zh-cn)下载 NCCl，安装后重新检测。

----------

##### 问题：多卡训练时启动失败，`Error：Out of all 4 Trainers`，如何处理？

+ 问题描述：多卡训练时启动失败，显示如下信息：

![图片](https://paddlepaddleimage.cdn.bcebos.com/faqimage%2Fbj-13d1b5df218cb40b0243d13450ab667f34aee2f7.png)

+ 报错分析：主进程发现一号卡（逻辑）上的训练进程退出了。

+ 解决方法：查看一号卡上的日志，找出具体的出错原因。`paddle.distributed.launch` 启动多卡训练时，设置 `--log_dir` 参数会将每张卡的日志保存在设置的文件夹下。

----------

##### 问题：训练时报错提示显存不足，如何解决？

+ 答复：可以尝试按如下方法解决：

1. 检查是当前模型是否占用了过多显存，可尝试减小`batch_size` ；

2. 开启以下三个选项：

```bash
#一旦不再使用即释放内存垃圾，=1.0 垃圾占用内存大小达到 10G 时，释放内存垃圾
export FLAGS_eager_delete_tensor_gb=0.0
#启用快速垃圾回收策略，不等待 cuda kernel 结束，直接释放显存
export FLAGS_fast_eager_deletion_mode=1
#该环境变量设置只占用 0%的显存
export FLAGS_fraction_of_gpu_memory_to_use=0
```

详细请参考官方文档[存储分配与优化](https://www.paddlepaddle.org.cn/documentation/docs/zh/dev_guides/api_contributing_guides/new_cpp_op_cn.html#xiancunyouhua) 调整相关配置。

此外，建议您使用[AI Studio 学习与 实训社区训练](https://aistudio.baidu.com/aistudio/index)，获取免费 GPU 算力，提升您的训练效率。

----------

##### 问题：如何提升模型训练时的 GPU 利用率？

+ 答复：有如下两点建议：

  1. 如果数据预处理耗时较长，可使用 DataLoader 加速数据读取过程，具体请参考 API 文档：[paddle.io.DataLoader](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/io/DataLoader_cn.html#dataloader)。

  2. 如果提高 GPU 计算量，可以增大`batch_size`，但是注意同时调节其他超参数以确保训练配置的正确性。

  以上两点均为比较通用的方案，其他的优化方案和模型相关，可参考官方模型库 [models](https://github.com/PaddlePaddle/models) 中的具体示例。

----------

##### 问题：如何处理变长 ID 导致程序内存占用过大的问题？

+ 答复：请先参考[显存分配与优化文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/dev_guides/api_contributing_guides/new_cpp_op_cn.html#xiancunyouhua) 开启存储优化开关，包括显存垃圾及时回收和 Op 内部的输出复用输入等。若存储空间仍然不够，建议：

  1. 降低 `batch_size` ；
  2. 对 index 进行排序，减少 padding 的数量。

----------

##### 问题：训练过程中如果出现不收敛的情况，如何处理？

+ 答复：不收敛的原因有很多，可以参考如下方式排查：

  1. 检查数据集中训练数据的准确率，数据是否有错误，特征是否归一化；
  2. 简化网络结构，先基于 benchmark 实验，确保在 baseline 网络结构和数据集上的收敛结果正确；
  3. 对于复杂的网络，每次只增加一个改动，确保改动后的网络正确；
  4. 检查网络在训练数据上的 Loss 是否下降；
  5. 检查学习率、优化算法是否合适，学习率过大会导致不收敛；
  6. 检查`batch_size`设置是否合适，`batch_size`过小会导致不收敛；
  7. 检查梯度计算是否正确，是否有梯度过大的情况，是否为`NaN`。

----------

##### 问题：Loss 为 NaN，如何处理？

+ 答复：可能由于网络的设计问题，Loss 过大（Loss 为 NaN）会导致梯度爆炸。如果没有改网络结构，但是出现了 NaN，可能是数据读取导致，比如标签对应关系错误。还可以检查下网络中是否会出现除 0，log0 的操作等。

----------


##### 问题：训练后的模型很大，如何压缩？

+ 答复：建议您使用飞桨模型压缩工具[PaddleSlim](https://www.paddlepaddle.org.cn/tutorials/projectdetail/489539)。PaddleSlim 是飞桨开源的模型压缩工具库，包含模型剪裁、定点量化、知识蒸馏、超参搜索和模型结构搜索等一系列模型压缩策略，专注于**模型小型化技术**。

----------

##### 问题：`load_inference_model`在加载预测模型时能否用`py_reader`读取？

+ 答复：目前`load_inference_model`加载进行的模型还不支持`py_reader`输入。

----------

##### 问题：预测时如何打印模型中每一步的耗时？

+ 答复：可以在设置 config 时使用`config.enable_profile()`统计预测时每个算子和数据搬运的耗时。对于推理 api 的使用，可以参考官网文档[Python 预测 API 介绍](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/05_inference_deployment/inference/python_infer_cn.html)。示例代码：
```python
# 设置 config:
def set_config(args):
    config = Config(args.model_file, args.params_file)
    config.disable_gpu()
    # enable_profile()打开后会统计每一步耗时
    config.enable_profile()
    config.switch_use_feed_fetch_ops(False)
    config.switch_specify_input_names(True)
    config.switch_ir_optim(False)
    return config
```
----------

##### 问题：模型训练时如何进行梯度裁剪？

+ 答复：设置 Optimizer 中的`grad_clip`参数值。

----------

##### 问题：静态图模型如何拿到某个 variable 的梯度？

+ 答复：飞桨提供以下三种方式，用户可根据需求选择合适的方法：

 1. 使用[paddle.static.Print()](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/static/Print_cn.html#print)接口，可以打印中间变量及其梯度。
 2. 将变量梯度名放到 fetch_list 里，通过[Executor.run()](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/static/Executor_cn.html#run)获取，一般 variable 的梯度名是 variable 的名字加上 "@GRAD"。
 3. 对于参数（不适用于中间变量和梯度），还可以通过[Scope.find_var()](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/static/global_scope_cn.html#global-scope)接口，通过变量名字查找对应的 tensor。

 后两个方法需要使用变量名，飞桨中变量的命名规则请参见[Name](https://www.paddlepaddle.org.cn/documentation/docs/zh/api_guides/low_level/program.html#api-guide-name) 。


```python
# paddlepaddle>=2.0
import paddle
import numpy as np

paddle.enable_static()
data = paddle.static.data('data', shape=[4, 2])
out = paddle.static.nn.fc(x=data, size=1, num_flatten_dims=1, name='fc')

loss = paddle.mean(out)
loss = paddle.static.Print(loss)  # 通过 Print 算子打印中间变量及梯度
opt = paddle.optimizer.SGD(learning_rate=0.01)
opt.minimize(loss)

exe = paddle.static.Executor()
exe.run(paddle.static.default_startup_program())
loss, loss_g, fc_bias_g = exe.run(
    paddle.static.default_main_program(),
    feed={'data': np.random.rand(4, 2).astype('float32')},
    fetch_list=[loss, loss.name + '@GRAD', 'fc.b_0@GRAD'])  # 通过将变量名加入到 fetch_list 获取变量

print(loss, loss_g, fc_bias_g)
print(paddle.static.global_scope().find_var('fc.b_0').get_tensor())  # 通过 scope.find_var 获取变量
```

----------

##### 问题：paddle 有对应 torch.masked_fill 函数 api 吗，还是需要自己实现？

+ 答复：由于框架设计上的区别，没有对应的 api，但是可以使用 [paddle.where](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/where_cn.html) 实现相同的功能。

```python
# paddlepaddle >= 2.0
import paddle

paddle.seed(123)
x = paddle.rand([3, 3], dtype='float32')
# Tensor(shape=[3, 3], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
#        [[0.00276479, 0.45899123, 0.96637046],
#         [0.66818708, 0.05855134, 0.33184195],
#         [0.34202638, 0.95503175, 0.33745834]])

mask = paddle.randint(0, 2, [3, 3]).astype('bool')
# Tensor(shape=[3, 3], dtype=bool, place=CUDAPlace(0), stop_gradient=True,
#        [[True , True , False],
#         [True , True , True ],
#         [True , True , True ]])

def masked_fill(x, mask, value):
    y = paddle.full(x.shape, value, x.dtype)
    return paddle.where(mask, y, x)

out = masked_fill(x, mask, 2)
# Tensor(shape=[3, 3], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
#        [[2.        , 2.        , 0.96637046],
#         [2.        , 2.        , 2.        ],
#         [2.        , 2.        , 2.        ]])
```

----------

##### 问题：在 paddle 中如何实现`torch.nn.utils.rnn.pack_padded_sequence`和`torch.nn.utils.rnn.pad_packed_sequence`这两个 API？

+ 答复：目前 paddle 中没有和上述两个 API 完全对应的实现。关于 torch 中这两个 API 的详细介绍可以参考知乎上的文章 [pack_padded_sequence 和 pad_packed_sequence](https://zhuanlan.zhihu.com/p/342685890) :
`pack_padded_sequence`的功能是将 mini-batch 数据进行压缩，压缩掉无效的填充值，然后输入 RNN 网络中；`pad_packed_sequence`则是把 RNN 网络输出的压紧的序列再填充回来，便于进行后续的处理。
在 paddle 中，大家可以在 GRU、LSTM 等 RNN 网络中输入含有填充值的 mini-batch 数据的同时传入对应的`sequence_length`参数实现上述等价功能，具体用法可以参考 [RNN](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/RNN_cn.html#rnn) 。

----------

##### 问题：paddle 是否有爱因斯坦求和（einsum）这个 api？

+ 答复：paddle 在 2.2rc 版本之后，新增了[paddle.einsum](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/einsum_cn.html#einsum)，在 develop 和 2.2rc 之后的版本中都可以正常使用。

----------


----------

##### 问题：[BatchNorm](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/BatchNorm_cn.html#batchnorm)在训练时加载预测时保存的模型参数时报错 AssertionError: Optimizer set error, batch_norm_1.w_0_moment_0 should in state dict.

+ 答复：BatchNorm 在 train 模式和 eval 模式下需要的变量有差别，在 train 模式下要求传入优化器相关的变量，在 eval 模式下不管是保存参数还是加载参数都是不需要优化器相关变量的，因此如果在 train 模式下加载 eval 模式下保存的 checkpoint，没有优化器相关的变量则会报错。如果想在 train 模式下加载 eval 模式下保存的 checkpoint 的话，用 ```paddle.load``` 加载进来参数之后，通过 ```set_state_dict``` 接口把参数赋值给模型，参考以下示例：

```python
import paddle

bn = paddle.nn.BatchNorm(3)
bn_param = paddle.load('./bn.pdparams')
bn.set_state_dict()
```

----------
