## 格式说明

> 提交代码前请参考[官网](https://www.paddlepaddle.org.cn/documentation/docs/zh/dev_guides/code_contributing_path_cn.html)安装 `pre-commit`，规范化代码格式。

整体的格式如下:

### [分类名称] api 全称

为了文档整体的一致性，我们统一了分类名称，分类名称需和下面保持一致。共分为 7 大类：

* 其中第１类又分为五种情况：`无参数`、`参数完全一致`、`仅参数名不一致`、`仅 paddle 参数更多`、`仅参数默认值不一致`。分类优先级依次递增，即如果参数名不一致，且 paddle 参数更多，则分为`仅 paddle 参数更多`。

* 第２类为`torch 参数更多`。如果 torch 和 paddle 都支持更多参数，统一写成`torch 参数更多`。

* 第３类为`参数用法不一致`。比如 所支持的参数类型不一致(是否可以是元组)、参数含义不一致。

* 第４类为 `组合替代实现` ，表示该 API 可以通过多个 API 组合实现。

* 第 5 类为 `用法不同：涉及上下文修改` ，表示涉及到上下文分析，需要修改其他位置的代码。

* 第 6 类为 `对应 API 不在主框架` 。例如 `torch.hamming_window` 对应 API 在 `paddlenlp` 中。

* 第 7 类为 `功能缺失` ，表示当前无该功能，则无需写差异分析文档，仅进行标注即可。

### [pytorch api 全称] (pytorch api 链接)

### [paddle api 全称] (paddle api 链接)

### 一句话描述部分

如果属于 `参数用法不一致` 分类，需要用 \` ` 加底色强调一下哪些参数用法不一致。

### 参数映射部分

参数映射表的左边是`PyTorch` 对应参数，右边是`Paddle`对应参数，表格参数顺序按 `PyTorch` 参数顺序来。

* 如果仅参数名不一致，需要在备注栏加一句`仅参数名不一致`。

* 如果 paddle 参数更多，需要在备注栏写下 paddle 应该如何设置此参数，或备注`PyTorch 无此参数， Paddle 保持默认即可`。

* 每个备注都需要`以句号结尾`。

### 转写示例部分
```python
# PyTorch 写法

# Paddle 写法

```

## 常用映射模板(供参考)

### 仅参数名不一致

两者功能一致且参数用法一致，仅参数名不一致，具体如下：

| PyTorch   | Paddle      | 备注                                                         |
| --------- | ----------- | ------------------------------------------------------------ |
| input     | x           | 表示输入的 Tensor ，仅参数名不一致。                         |
| other     | y           | 表示输入的 Tensor ，仅参数名不一致。                         |
| dim       | axis        | 表示进行运算的轴，仅参数名不一致。                           |
| dtype     | dtype       | 表示数据类型。                                               |
| size      | shape       | 表示输出形状大小。                                           |
| n         | num_rows    | 生成 2-D Tensor 的行数，仅参数名不一致。                     |
| m         | num_columns | 生成 2-D Tensor 的列数， 仅参数名不一致。                    |
| start_dim | start_axis  | 表示 flatten 展开的起始维度。                                |
| end_dim   | stop_axis   | 表示 flatten 展开的结束维度。                                |
| ndarray   | data        | 表示需要转换的数据， PyTorch 只能传入 numpy.ndarray ， Paddle 可以传入 scalar 、 list 、 tuple 、 numpy.ndarray 、 paddle.Tensor 。 |


### 仅 Paddle 参数更多

其中 Paddle 相比 PyTorch 支持更多其他参数，具体如下：

| PyTorch | Paddle        | 备注                                                         |
| ------- | ------------- | ------------------------------------------------------------ |
| -       | axis          | 指定进行运算的轴， Pytorch 无此参数， Paddle 保持默认即可。  |
| -       | keepdim       | 是否在输出 Tensor 中保留减小的维度， Pytorch 无此参数， Paddle 保持默认即可。 |
| -       | dtype         | 输出 Tensor 的数据类型， Pytorch 无此参数， Paddle 保持默认即可。 |
| -       | dtype         | 表示数据类型， PyTorch 无此参数， Paddle 保持默认即可。      |
| -       | place         | 表示 Tensor 存放位置， PyTorch 无此参数， Paddle 需设置为 paddle.CPUPlace()。 |
| -       | stop_gradient | 表示是否阻断梯度传导， PyTorch 无此参数， Paddle 保持默认即可。 |

### torch 参数更多

其中 Pytorch 相比 Paddle 支持更多其他参数，具体如下：

| PyTorch       | Paddle | 备注                                                         |
| ------------- | ------ | ------------------------------------------------------------ |
| input         | x      | 表示输入的 Tensor ，仅参数名不一致。                         |
| out           | -      | 表示输出的 Tensor ， Paddle 无此参数，需要进行转写。         |
| *size         | shape  | 表示输出形状大小， PyTorch 是多个元素， Paddle 是列表或元组，需要进行转写。 |
| layout        | -      | 表示布局方式， Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。 |
| device        | -      | 表示 Tensor 存放设备位置，Paddle 无此参数，需要进行转写。    |
| requires_grad | -      | 表示是否计算梯度， Paddle 无此参数，需要进行转写。           |
| memory_format | -      | 表示内存格式， Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。 |
| pin_memeory   | -      | 表示是否使用锁页内存， Paddle 无此参数，需要进行转写。       |
| generator     | -      | 用于采样的伪随机数生成器， Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。 |
| size_average  | -      | PyTorch 已弃用， Paddle 无此参数，需要转写。                  |
| reduce        | -      | PyTorch 已弃用， Paddle 无此参数，需要转写。                  |

转写示例

#### *size：输出形状大小
```python
# Pytorch 写法
torch.empty(3, 5)

# Paddle 写法
paddle.empty([3, 5])
```


#### out：指定输出
```python
# Pytorch 写法
torch.abs([-3, -5], out=y)

# Paddle 写法
y = paddle.abs([-3, -5])
```

#### device: Tensor 的设备
```python
# Pytorch 写法
torch.zeros_like(x, device=torch.device('cpu'))

# Paddle 写法
y = paddle.zeros_like(x)
y.cpu()
```

#### requires_grad：是否求梯度
```python
# Pytorch 写法
x = torch.zeros_like(x, requires_grad=True)

# Paddle 写法
x = paddle.zeros_like(x)
x.stop_gradient = False
```
#### pin_memory：是否分配到固定内存上
```python
# Pytorch 写法
x = torch.empty_like((2, 3), pin_memory=True)

# Paddle 写法
x = paddle.empty_like([2, 3]).pin_memory()
```

#### size_average
```python
# Pytorch 的 size_average、reduce 参数转为 Paddle 的 reduction 参数
if size_average is None:
    size_average = True
if reduce is None:
    reduce = True

if size_average and reduce:
    reduction = 'mean'
elif reduce:
    reduction = 'sum'
else:
    reduction = 'none'
```

### 参数用法不一致

| PyTorch | Paddle        | 备注                                                         |
| ------- | ------------- | ------------------------------------------------------------ |
|  *tensors       |  inputs    | 一组输入 Tensor ， Pytorch 参数 tensors 为可变参, Paddle 参数 inputs 为 list(Tensor) 或 tuple(Tensor) 的形式。   |

转写示例

#### *tensors: 一组输入 Tensor
```python
# Pytorch 写法
torch.broadcast_tensors(x, y)

# Paddle 写法
paddle.broadcast_tensors([x, y])
```

#### affine：是否进行反射变换

```python
affine=False 时，表示不更新：

# PyTorch 写法
m = torch.nn.BatchNorm1D(24, affine=False)

# Paddle 写法
weight_attr = paddle.ParamAttr(learning_rate=0.0)
bias_attr = paddle.ParamAttr(learning_rate=0.0)
m = paddle.nn.BatchNorm1D(24, weight_attr=weight_attr, bias_attr=bias_attr)

affine=True 时，表示更新：

# PyTorch 写法
m = torch.nn.BatchNorm1D(24)

# Paddle 写法
m = paddle.nn.BatchNorm1D(24)
```

### 用法不同：涉及上下文修改

其中 Pytorch 与 Paddle 对该 API 的设计思路与⽤法不同，需要分析上下⽂并联动修改：

| PyTorch | PaddlePaddle | 备注 |
| ------- | ------------ | ---- |
| parameters |  -  | 表示要操作的 Tensor， Pytorch 属于原位操作， PaddlePaddle ⽆此参数，需要实例化之后在 optimizer 中设置才可以使⽤。需要上下⽂分析与联动修改。|
| clip_value |  max |  表示裁剪梯度的范围，范围为 [-clip_value, clip_vale] ； PaddlePaddle 的 max 参数可实现该参数功能，直接设置为与 clip_value ⼀致。|
| - | min | 表示裁剪梯度的最⼩值， PyTorch ⽆此参数， Paddle 保持默认即可。 |

### 转写示例

```python
# torch ⽤法
net = Model()
sgd = torch.optim.SGD(net.parameters(), lr=0.1)
for i in range(10):
 loss = net(x)
 loss.backward()
 torch.nn.utils.clip_grad_value_(net.parameters(), 1.)
 sgd.step()

# paddle ⽤法
net = Model()
sgd = paddle.optim.SGD(net.parameters(), lr=0.1,
grad_clip=paddle.nn.ClipGradByValue(), 1.)
for i in range(10):
 loss = net(x)
 loss.backward()
 sgd.step()
```
