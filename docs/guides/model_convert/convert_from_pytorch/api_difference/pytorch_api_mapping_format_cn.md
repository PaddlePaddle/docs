> 提交代码前请参考[官网](https://www.paddlepaddle.org.cn/documentation/docs/zh/dev_guides/code_contributing_path_cn.html)安装 `pre-commit`，规范化代码格式。

请严格根据此格式规范来新增《API 映射关系》，不符合规范的文档将不予合入，具体如下:

# API 映射关系 - 格式规范

### [分类名称] api 全称

为了文档整体的一致性，我们统一了 API 映射关系的分类名称，共分为 7 大类：
> 注：第 1~3 类均为 API 层面一对一映射，根据参数层面的映射关系将其细分为三类。

* 第 1 类又分为五种情况：`无参数`、`参数完全一致`、`仅参数名不一致`、`paddle 参数更多`、`参数默认值不一致`。
> 注：分类优先级依次递增，即如果同时 `参数名不一致` + `paddle 参数更多`，则写成后者 `paddle 参数更多` 。

* 第 2 类为 `torch 参数更多`。如果 torch 和 paddle 都支持更多参数，统一写成`torch 参数更多`。

* 第 3 类又分为三种情况：`输入参数类型不一致`、`输入参数用法不一致`、`返回参数类型不一致`。
> 注意：这里的**不一致**都是从 torch 的角度来看，如果 paddle 可涵盖 torch，即 torch 是 paddle 的功能子集，则认定为一致（例如：torch 参数仅支持 list，paddle 参数支持 list/tuple），反之才认定为不一致。

* 第 4 类为 `组合替代实现` ，表示该 API 没有可直接对应的 API，需要通过多个 API 组合实现。

* 第 5 类为 `涉及上下文修改` ，表示涉及到上下文的分析，需要修改其他位置的代码。
> 举例：所有的 `torch.optim.lr_scheduler.*`、`torch.nn.init.*`、`torch.nn.utils.clip*` 都为该类。此类 API 由于两者在设计上具有较大的差异，需要对上下文进行分析，涉及到上下文代码的联动修改。

* 第 6 类为 `可删除` 。表示可直接删除该 API，则无需写差异分析文档，仅标注即可

* 第 7 类为 `功能缺失` ，表示 Paddle 当前无对应 API，则无需写差异分析文档，仅标注即可。

> 注意：
> 1. 分类优先级依次递增，即如果同时 `第 2 类：torch 参数更多` 与 `第 3 类：参数不一致` ，则写成后者 `第 3 类：参数不一致` 。
> 2. 所有的 Paddle API 无需关注 `name` 参数，直接忽略即可。
> 3. 将类成员 API 映射为非类成员 API，则无需对比第一个参数。例如将 `torch.Tensor.outer(vec2)` 映射为 `paddle.outer(x, y)`，则忽略 paddle 的第一个参数，从 torch 的 `vec2` 和 paddle 的 `y` 开始对比。

### [pytorch api 全称] (pytorch api 链接)

```python
PyTorch API 签名
```

### [paddle api 全称] (paddle api 链接)

```python
Paddle API 签名
```

**一句话总结**。整体概述总结两个 API 的差异。例如 `第 3 类：参数不一致` ，需要简述下有哪些不一致的地方。在描写参数时，需要用 \` ` 来加深其底色。

### 参数映射

参数映射表的左边是`PyTorch` 对应参数，右边是`Paddle`对应参数，表格参数顺序按 `PyTorch` 参数顺序来。

* 如果 `无参数`，则没有参数映射这一栏。

* 如果 `参数完全一致`，无需转写示例。

* 如果 `仅参数名不一致`，无需转写示例，需要在备注栏里对该参数加一句 `仅参数名不一致`。

* 如果 `paddle 参数更多`，无需转写示例，需要在备注栏加一句 `PyTorch 无此参数，（Paddle 应如何设置此参数）` 。如果默认无影响，则写 `PyTorch 无此参数，Paddle 保持默认即可`。

* 如果 `参数默认值不一致`，无需转写示例，需要在备注栏里加一句 `与 PyTorch 默认值不同，（Paddle 应如何设置此参数）` 。

* 如果 `torch 参数更多`，对每个 torch 多的参数都需要转写示例，需要在备注栏里加一句 `Paddle 无此参数，需要转写` ；如确实无法转写，需要在备注里写 `Paddle 无此参数，暂无转写方式` ；若可直接删除，则需要写 `Paddle 无此参数，一般对网络训练结果影响不大，可直接删除` 。

* 如果 `参数不一致`，对每个不一致的参数都需要转写示例，需要在备注栏里写 `（说明不一致的用法），需要转写`；如确实无法转写，需要在备注里写 `（说明不一致的用法），暂无转写方式`。

* 每个备注都需要`以句号结尾`。

### 转写示例

**除第 1 类 API 映射关系较为简单，无需写转写示例，其他类 API 都需要写转写示例，否则需说明：Paddle 暂无转写方式。**

转写示例需要写得精简和一目了然。一般情形下只需写两行代码，无需打印各种结果，并且要保证转写前后的输出结果是一致的。另外需要先描述下待写的是该 torch api 的哪个参数及其功能。

#### 参数名 1：参数功能 1
```python
# PyTorch 写法
torch.xxx()

# Paddle 写法
paddle.xxx()
```

#### 参数名 2：参数功能 2
```python
# PyTorch 写法
torch.xxx()

# Paddle 写法
paddle.xxx()
```

--------------

# API 映射关系 - 模板

## 模板 1

### [ 无参数 ] torch.Tensor.t

### [torch.Tensor.t](https://pytorch.org/docs/stable/generated/torch.Tensor.t.html#torch.Tensor.t)

```python
torch.Tensor.t()
```

### [paddle.Tensor.t](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#t-name-none)

```python
paddle.Tensor.t()
```

两者功能一致，无参数。

## 模板 2

### [ 参数完全一致 ] torch.Tensor.clip

### [torch.Tensor.clip](https://pytorch.org/docs/stable/generated/torch.Tensor.clip.html?highlight=clip#torch.Tensor.clip)

```python
torch.Tensor.clip(min=None, max=None)
```

### [paddle.Tensor.clip](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/Tensor_cn.html#clip-min-none-max-none-name-none)

```python
paddle.Tensor.clip(min=None, max=None, name=None)
```

两者功能一致，参数完全一致，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                                               |
|---------|--------------| -------------------------------------------------- |
| min     | min          | 裁剪的最小值，输入中小于该值的元素将由该元素代替。            |
| max     | max          | 裁剪的最大值，输入中大于该值的元素将由该元素代替。            |


## 模板 3

### [ 仅参数名不一致 ] torch.dist

### [torch.dist](https://pytorch.org/docs/stable/generated/torch.dist.html?highlight=dist#torch.dist)(仅作为示例)

```python
torch.dist(input,
           other,
           p=2)
```

### [paddle.dist](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/dist_cn.html#dist)(仅作为示例)

```python
paddle.dist(x,
            y,
            p=2)
```

两者功能一致且参数用法一致，仅参数名不一致，具体如下：

### 参数映射

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


## 模板 4

### [ paddle 参数更多 ] torch.ZeroPad2d

### [torch.nn.ZeroPad2d](https://pytorch.org/docs/stable/generated/torch.nn.ZeroPad2d.html?highlight=zeropad#torch.nn.ZeroPad2d)(仅作为示例)

```python
torch.nn.ZeroPad2d(padding)
```

### [paddle.nn.ZeroPad2D](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/ZeroPad2D_cn.html)(仅作为示例)

```python
paddle.nn.ZeroPad2D(padding,
                    data_format='NCHW',
                    name=None)
```

Paddle 相比 PyTorch 支持更多其他参数，具体如下：

### 参数映射

| PyTorch | Paddle        | 备注                                                         |
| ------- | ------------- | ------------------------------------------------------------ |
| -       | axis          | 指定进行运算的轴， PyTorch 无此参数， Paddle 保持默认即可。  |
| -       | keepdim       | 是否在输出 Tensor 中保留减小的维度， PyTorch 无此参数， Paddle 保持默认即可。 |
| -       | dtype         | 输出 Tensor 的数据类型， PyTorch 无此参数， Paddle 保持默认即可。 |
| -       | dtype         | 表示数据类型， PyTorch 无此参数， Paddle 保持默认即可。      |
| -       | place         | 表示 Tensor 存放位置， PyTorch 无此参数， Paddle 需设置为 paddle.CPUPlace()。 |
| -       | stop_gradient | 表示是否阻断梯度传导， PyTorch 无此参数， Paddle 保持默认即可。 |


## 模板 5

## [ 参数默认值不一致 ] torch.linalg.svd

### [torch.linalg.svd](https://pytorch.org/docs/stable/generated/torch.linalg.svd.html?highlight=svd#torch.linalg.svd)

```python
torch.linalg.svd(A, full_matrices=True)
```

### [paddle.linalg.svd](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/linalg/svd_cn.html)

```python
paddle.linalg.svd(x, full_matrices=False, name=None)
```

两者功能一致且参数用法一致，参数默认值不一致，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| A             | x           | 输入 Tensor，仅参数名不一致。               |
| full_matrices | full_matrices  | 是否计算完整的 U 和 V 矩阵，PyTorch 为 True，Paddle 为 False，Paddle 需设置为与 PyTorch 一致。  |


## 模板 6

### [ torch 参数更多 ] torch.abs

### [torch.abs](https://pytorch.org/docs/stable/generated/torch.abs.html?highlight=abs#torch.abs)(仅作为示例)

```python
torch.abs(input,
          *,
          out=None)
```

### [paddle.abs](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/abs_cn.html#abs)(仅作为示例)

```python
paddle.abs(x,
           name=None)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch       | Paddle | 备注                                                         |
| ------------- | ------ | ------------------------------------------------------------ |
| input         | x      | 表示输入的 Tensor ，仅参数名不一致。                         |
| out           | -      | 表示输出的 Tensor ，Paddle 无此参数，需要转写。         |
| *size         | shape  | 表示输出形状大小， PyTorch 是多个元素，Paddle 是列表或元组，需要转写。 |
| layout        | -      | 表示布局方式， Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。 |
| device        | -      | 表示 Tensor 存放设备位置，Paddle 无此参数，需要转写。    |
| requires_grad | -      | 表示是否计算梯度， Paddle 无此参数，需要转写。           |
| memory_format | -      | 表示内存格式， Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。 |
| pin_memeory   | -      | 表示是否使用锁页内存， Paddle 无此参数，需要转写。       |
| generator     | -      | 用于采样的伪随机数生成器， Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。 |
| size_average  | -      | PyTorch 已弃用， Paddle 无此参数，需要转写。                  |
| reduce        | -      | PyTorch 已弃用， Paddle 无此参数，需要转写。                  |
| async_op      | -      | 是否异步操作，Paddle 无此参数，暂无转写方式。                   |
| antialias     | -      | 是否使用 anti-aliasing，Paddle 无此参数，暂无转写方式。        |

### 转写示例
#### size：输出形状大小
```python
# PyTorch 写法
torch.empty(3, 5)

# Paddle 写法
paddle.empty([3, 5])
```

#### out：指定输出
```python
# PyTorch 写法
torch.abs([-3, -5], out=y)

# Paddle 写法
paddle.assign(paddle.abs([-3, -5]), y)
```

#### device: Tensor 的设备
```python
# PyTorch 写法
torch.zeros_like(x, device=torch.device('cpu'))

# Paddle 写法
y = paddle.zeros_like(x)
y.cpu()
```

#### requires_grad：是否求梯度
```python
# PyTorch 写法
x = torch.zeros_like(x, requires_grad=True)

# Paddle 写法
x = paddle.zeros_like(x)
x.stop_gradient = False
```
#### pin_memory：是否分配到固定内存上
```python
# PyTorch 写法
x = torch.empty_like((2, 3), pin_memory=True)

# Paddle 写法
x = paddle.empty_like([2, 3]).pin_memory()
```

#### size_average：做 reduce 的方式
```python
# PyTorch 的 size_average、reduce 参数转为 Paddle 的 reduction 参数
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


## 模板 7

### [ 输入参数用法不一致 ] torch.transpose

### [torch.transpose](https://pytorch.org/docs/stable/generated/torch.transpose.html?highlight=transpose#torch.transpose)(仅作为示例)

```python
torch.transpose(input,
                dim0,
                dim1)
```

### [paddle.transpose](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/transpose_cn.html#transpose)(仅作为示例)

```python
paddle.transpose(x,
                 perm,
                 name=None)
```

PyTorch 的 `tensors` 参数与 Paddle 的 `inputs` 参数用法不同，具体如下：

### 参数映射

| PyTorch | Paddle        | 备注                                                         |
| ------- | ------------- | ------------------------------------------------------------ |
|*tensors |  inputs    | 一组输入 Tensor，PyTorch 的 tensors 为可变参数，Paddle 的 inputs 为 list(Tensor) 或 tuple(Tensor) 用法，需要转写。   |
| 返回值    | 返回值  | 返回参数类型不一致, PyTorch 返回 torch.ByteTensor，Paddle 返回 GeneratorState 对象，暂无转写方式。 |


### 转写示例
#### *tensors: 一组输入 Tensor，可变参数用法
```python
# PyTorch 写法
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

## 模板 8

### [ 组合替代实现 ] torch.addcmul

### [torch.addcmul](https://pytorch.org/docs/master/generated/torch.addcmul.html#torch.addcmul)

```python
torch.addcmul(input, tensor1, tensor2, *, value=1, out=None)
```

用于实现矩阵 `tensor1` 与矩阵 `tensor2` 相乘，再加上输入 `input` ，公式为：

$ out =  input + value *  tensor1 * tensor2 $

PaddlePaddle 目前无对应 API，可使用如下代码组合替代实现:

### 转写示例

```python
# PyTorch 写法
torch.addcmul(input, tensor1, tensor2, value=value)

# Paddle 写法
paddle.add(input, value * tensor1 * tensor2)
```

## 模板 9

### [ 涉及上下文修改 ] torch.nn.utils.clip_grad_value_

### [torch.nn.utils.clip_grad_value_](https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_value_.html?highlight=clip_grad_value_#torch.nn.utils.clip_grad_value_)(仅作为示例)

```python
torch.nn.utils.clip_grad_value_(parameters,
                                clip_value)
```

### [paddle.nn.ClipGradByValue](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/ClipGradByValue_cn.html#clipgradbyvalue)(仅作为示例)

```python
paddle.nn.ClipGradByValue(max,
                          min=None)
```

其中 PyTorch 与 Paddle 对该 API 的设计思路与⽤法不同，需要分析上下⽂并联动修改：

### 参数映射

| PyTorch | PaddlePaddle | 备注 |
| ------- | ------------ | ---- |
| parameters |  -  | 表示要操作的 Tensor， PyTorch 属于原位操作， PaddlePaddle ⽆此参数，需要实例化之后在 optimizer 中设置才可以使⽤。需要上下⽂分析与联动修改。|
| clip_value |  max |  表示裁剪梯度的范围，范围为 [-clip_value, clip_vale] ； PaddlePaddle 的 max 参数可实现该参数功能，直接设置为与 clip_value 一致。|
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
