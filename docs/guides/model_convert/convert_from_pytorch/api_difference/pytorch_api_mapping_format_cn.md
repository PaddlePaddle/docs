> 提交代码前请参考[官网](https://www.paddlepaddle.org.cn/documentation/docs/zh/dev_guides/code_contributing_path_cn.html)安装 `pre-commit`，规范化代码格式。

请严格根据此格式规范来新增《API 映射关系》，不符合规范的文档将不予合入，具体如下:

# API 映射关系文档 - 规范

### [分类名称] api 全称

由于 API 映射关系的复杂性，为了保证文档格式的规范性，我们将所有 API 映射关系分为 12 类，并制定了统一的 **分类名称**：
1. 无参数
2. 参数完全一致
3. 仅参数名不一致
4. paddle 参数更多
5. 参数默认值不一致
6. torch 参数更多
7. 输入参数用法不一致
8. 输入参数类型不一致
9. 返回参数类型不一致
10. 组合替代实现
11. 可删除
12. 功能缺失

> 注意：
> 1. 分类的优先级依次递增，例如：如果同时 `仅参数名不一致` + `paddle 参数更多`，则分类为后者 `paddle 参数更多` ，如果同时 `paddle 参数更多` + `torch 参数更多`，则分类为后者 `torch 参数更多`。
> 2. `输入参数用法不一致`、`输入参数类型不一致`、`返回参数类型不一致` 其中的**不一致**都是从 torch 的角度来看，只要 torch 能被 paddle 全覆盖，则将其视作一致（例如：torch 参数仅支持 list 用法，而 paddle 参数支持 list/tuple 用法，则视作用法一致），如果 torch 无法被 paddle 全覆盖，才认定为 **不一致**。
> 3. `可删除` 表示转写时可直接删除该 API，并不会对代码运行结果有影响，无需写映射文档，仅标注即可。`功能缺失` 表示 Paddle 当前无对应 API 功能，则无需写映射文档，仅标注即可。
> 4. 所有的 Paddle API 无需关注 `name` 参数，直接忽略即可。
> 5. 将类成员 API 映射为非类成员 API，则无需对比第一个参数。例如将 `torch.Tensor.outer(vec2)` 映射为 `paddle.outer(x, y)`，则忽略 paddle 的第一个参数 `x` ，直接从 torch 的 `vec2` 和 paddle 的 `y` 开始对比参数。

### [pytorch api 全称] (pytorch api 链接)

```python
PyTorch API 签名
```

### [paddle api 全称] (paddle api 链接)

```python
Paddle API 签名
```

**一句话总结**。整体概述总结两个 API 的差异。例如 `输入参数用法不一致` ，需要简述下有参数哪些用法不一致的地方。在描写参数时，需要用 \` ` 来加深其底色。

### 参数映射

参数映射以表格的形式呈现，表格的第 1 列是`PyTorch` 所有参数，第 2 列是`Paddle`对应参数，表格顺序按第 1 列 `PyTorch` 的参数顺序来。

1. **无参数**：无需参数映射与转写示例。

2. **参数完全一致**：无需转写示例。

3. **仅参数名不一致**：无需转写示例，但需要在备注列里注明哪些参数 `仅参数名不一致`。

4. **paddle 参数更多**：无需转写示例，但需要在备注列里注明 `PyTorch 无此参数，[Paddle 应如何设置此参数]`。如果无需特殊设置，则写 `PyTorch 无此参数，Paddle 保持默认即可`。

5. **参数默认值不一致**：无需转写示例，但需要在备注列里注明 `参数默认值不一致，[Paddle 应如何设置此参数，设置为多少]`。

6. **torch 参数更多**：对每个 torch 多的参数都需要**转写示例**，同时需要在备注列里注明 `Paddle 无此参数，需要转写` ；如确实无法转写，则注明 `Paddle 无此参数，暂无转写方式` ；若可直接删除，则需要写 `Paddle 无此参数，一般对网络训练结果影响不大，可直接删除` 。

7. **输入参数用法不一致、输入参数类型不一致、返回参数类型不一致**：对每个不一致的参数都需要**转写示例**，同时需要在备注列里注明 `[不一致的用法说明]，需要转写`；如确实无法转写，需要在备注里写 `[不一致的用法说明]，暂无转写方式`。

8. **组合替代实现**：无需参数映射，需要写 **转写示例** 。

9. 每个备注都需要`以句号结尾`。

### 转写示例

转写示例需要写得精简和一目了然。一般情形下只需写两行代码，无需打印各种结果，需要保证转写前后的输出结果是一致的。另外需要先描述下转写的是 torch api 的哪个参数及其功能。

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

是否需要 **参数映射** 与 **转写示例** 可查阅表格：

第 1、10 类不需要参数映射，其他类均需要写参数映射。第 1-5 类不需要转写示例，第 6-10 类需要写转写示例。第 11-12 类则无需新建文档，仅标注即可。

| 分类序号 |    分类名称            |    参数映射      |    转写示例      |
| ------- | -----------------    | --------------- | --------------- |
| 1       | 无参数                | ❌              |  ❌              |
| 2       | 参数完全一致           | ✅              |  ❌              |
| 3       | 仅参数名不一致         | ✅              |  ❌              |
| 4       | paddle 参数更多       | ✅              |  ❌              |
| 5       | 参数默认值不一致       | ✅              |  ❌              |
| 6       | torch 参数更多        | ✅              |  ✅              |
| 7       | 输入参数用法不一致      | ✅              |  ✅              |
| 8       | 输入参数类型不一致      | ✅              |  ✅              |
| 9       | 返回参数类型不一致      | ✅              |  ✅              |
| 10      | 组合替代实现           | ❌              |  ✅              |

--------------------------------------------------------

# API 映射关系文档 - 模板


## 分类 1：无参数

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


## 分类 2：参数完全一致

### [ 参数完全一致 ] torch.Tensor.clip

### [torch.Tensor.clip](https://pytorch.org/docs/stable/generated/torch.Tensor.clip.html?highlight=clip#torch.Tensor.clip)

```python
torch.Tensor.clip(min=None, max=None)
```

### [paddle.Tensor.clip](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#clip-min-none-max-none-name-none)

```python
paddle.Tensor.clip(min=None, max=None, name=None)
```

两者功能一致，参数完全一致，具体如下：
### 参数映射

| PyTorch | PaddlePaddle | 备注                                               |
|---------|--------------| -------------------------------------------------- |
| min     | min          | 裁剪的最小值，输入中小于该值的元素将由该元素代替。            |
| max     | max          | 裁剪的最大值，输入中大于该值的元素将由该元素代替。            |



## 分类 3：仅参数名不一致

### [ 仅参数名不一致 ]torch.dist

### [torch.dist](https://pytorch.org/docs/stable/generated/torch.dist.html?highlight=dist#torch.dist)

```python
torch.dist(input,
           other,
           p=2)
```

### [paddle.dist](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/dist_cn.html#dist)

```python
paddle.dist(x,
            y,
            p=2)
```

两者功能一致且参数用法一致，仅参数名不一致，具体如下：
### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input         | x            | 表示输入的 Tensor ，仅参数名不一致。  |
| other         | y            | 表示输入的 Tensor ，仅参数名不一致。  |
| p             | p            | 表示需要计算的范数 |



## 分类 4：paddle 参数更多

### [ paddle 参数更多 ]torch.nn.ZeroPad2d
### [torch.nn.ZeroPad2d](https://pytorch.org/docs/stable/generated/torch.nn.ZeroPad2d.html?highlight=zeropad#torch.nn.ZeroPad2d)

```python
torch.nn.ZeroPad2d(padding)
```

### [paddle.nn.ZeroPad2D](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/ZeroPad2D_cn.html)

```python
paddle.nn.ZeroPad2D(padding,
                    data_format='NCHW',
                    name=None)
```

其中 Paddle 相比 PyTorch 支持更多其他参数，具体如下：
### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| padding       | padding      | 表示填充大小。                             |
| -             | data_format  | 指定输入的 format， PyTorch 无此参数， Paddle 保持默认即可。 |



## 分类 5：参数默认值不一致

### [ 参数默认值不一致 ]torch.linalg.diagonal
### [torch.linalg.diagonal](https://pytorch.org/docs/stable/generated/torch.linalg.diagonal.html#torch.linalg.diagonal)

```python
torch.linalg.diagonal(A, *, offset=0, dim1=-2, dim2=-1)
```

### [paddle.diagonal](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/diagonal_cn.html#diagonal)

```python
paddle.diagonal(x,
                offset=0,
                axis1=0,
                axis2=1,
                name=None)
```

两者功能一致且参数用法一致，参数默认值不一致，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle | 备注                              |
| ------------- | ------------ | -------------------------------- |
| A             | x            | 表示输入的 Tensor ，仅参数名不一致。  |
| offset        | offset       | 表示对角线偏移量。                  |
| dim1          | axis1        | 获取对角线的二维平面的第一维，参数默认值不一致。PyTorch 默认为`-2`，Paddle 默认为`0`，Paddle 需设置为与 PyTorch 一致。  |
| dim2          | axis2        | 获取对角线的二维平面的第二维，参数默认值不一致。PyTorch 默认为`-1`，Paddle 默认为`1`，Paddle 需设置为与 PyTorch 一致。  |


## 分类 6：torch 参数更多

### [ torch 参数更多 ] torch.abs

### [torch.abs](https://pytorch.org/docs/stable/generated/torch.abs.html?highlight=abs#torch.abs)

```python
torch.abs(input,
          *,
          out=None)
```

### [paddle.abs](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/abs_cn.html#abs)

```python
paddle.abs(x,
           name=None)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：（注：这里额外列举了一些其他常见 Pytorch 的参数的转写方式，与 torch.abs 无关）

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


## 分类 7：输入参数用法不一致

## [ 输入参数用法不一致 ] torch.transpose

### [torch.transpose](https://pytorch.org/docs/stable/generated/torch.transpose.html?highlight=transpose#torch.transpose)

```python
torch.transpose(input,
                dim0,
                dim1)
```

### [paddle.transpose](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/transpose_cn.html#transpose)

```python
paddle.transpose(x,
                 perm,
                 name=None)
```

其中 PyTorch 的 `dim0、dim1` 与 Paddle 用法不一致，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| <font color='red'>input</font>         | <font color='red'>x</font>            | 输入 Tensor。                                       |
| <font color='red'>dim0</font>          | -            | PyTorch 转置的第一个维度，Paddle 无此参数，需要转写。                    |
| <font color='red'>dim1</font>          | -            | PyTorch 转置的第二个维度，Paddle 无此参数，需要转写。                    |
| -             | <font color='red'>perm</font>         | Paddle 可通过 perm 参数，等价的实现 torch 的 dim0、dim1 的功能。|


### 转写示例

#### dim0、dim1 参数： 转置的维度设置
``` python
# PyTorch 写法:
torch.transpose(x, dim0=0, dim1=1)

# Paddle 写法:
paddle.transpose(x, perm=[1, 0, 2])

# 注：x 为 3D Tensor
```


## 分类 8：输入参数类型不一致

### [ 输入参数类型不一致 ]torch.broadcast_tensors

### [torch.broadcast_tensors](https://pytorch.org/docs/stable/generated/torch.broadcast_tensors.html?highlight=broadcast_tensors#torch.broadcast_tensors)

```python
torch.broadcast_tensors(*tensors)
```

### [paddle.broadcast_tensors](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/broadcast_tensors_cn.html#broadcast-tensors)

```python
paddle.broadcast_tensors(inputs,
                         name=None)
```

两者功能一致但参数类型不一致，具体如下：
### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| *tensors      | inputs       | 一组输入 Tensor ， PyTorch 参数 tensors 为可变参, Paddle 参数 inputs 为 list(Tensor) 或 tuple(Tensor) 的形式。   |


### 转写示例
#### *tensors: 一组输入 Tensor
```python
# PyTorch 写法
torch.broadcast_tensors(x, y)

# Paddle 写法
paddle.broadcast_tensors([x, y])
```


## 分类 9：返回参数类型不一致

### [ 返回参数类型不一致 ]torch.equal
### [torch.equal](https://pytorch.org/docs/stable/generated/torch.equal.html?highlight=equal#torch.equal)

```python
torch.equal(input,
            other)
```

### [paddle.equal_all](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/equal_all_cn.html#equal-all)

```python
paddle.equal_all(x,
                 y,
                 name=None)
```

两者功能一致但返回参数类型不同，具体如下：
### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input         | x            | 表示输入的 Tensor，仅参数名不一致。  |
| other         | y            | 表示输入的 Tensor，仅参数名不一致。  |
| 返回值         | 返回值        | PyTorch 返回 bool 类型，Paddle 返回 0-D bool Tensor，需要转写。|

### 转写示例
#### 返回值
``` python
# PyTorch 写法
out = torch.equal(x, y)

# Paddle 写法
out = paddle.equal_all(x, y)
out = out.item()
```


## 分类 10：组合替代实现

### [ 组合替代实现 ]torch.aminmax

### [torch.aminmax](https://pytorch.org/docs/stable/generated/torch.aminmax.html#torch.aminmax)

```python
torch.aminmax(input, *, dim=None, keepdim=False, out=None)
```
Paddle 无此 API，需要组合实现。

### 转写示例

```python
# PyTorch 写法
y = torch.aminmax(input, dim=-1, keepdim=True)

# Paddle 写法
y = tuple([paddle.amin(input, axis=-1, keepdim=True), paddle.amax(input, axis=-1, keepdim=True)])
```
