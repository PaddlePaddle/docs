## [torch 参数更多]torch.empty

###  [torch.empty](https://pytorch.org/docs/1.13/generated/torch.empty.html?highlight=empty#torch.empty)

```python
torch.empty(*size,
            *,
            out=None,
            dtype=None,
            layout=torch.strided,
            device=None,
            requires_grad=False,
            pin_memory=False,
            memory_format=torch.contiguous_format)
```

###  [paddle.empty](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/empty_cn.html)

```python
paddle.empty(shape,
             dtype=None,
             name=None)
```

其中 Pytorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                         |
| :------------ | :----------- | :----------------------------------------------------------- |
| *size         | shape        | 表示输出形状大小， PyTorch 是多个元素， Paddle 是列表或元组，需要进行转写。 |
| out           | -            | 表示输出的 Tensor，Paddle 无此参数，需要进行转写。           |
| dtype         | dtype        | 表示输出 Tensor 类型。                                       |
| layout        | -            | 表示布局方式，Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。 |
| device        | -            | 表示 Tensor 存放设备位置，Paddle 无此参数，需要进行转写。    |
| requires_grad | -            | 表示是否计算梯度，Paddle 无此参数，需要进行转写。            |
| pin_memory    | -            | 表示是否使用锁页内存， Paddle 无此参数，需要进行转写。       |
| memory_format | -            | 表示内存格式， Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。 |

### 转写示例

#### size：输出形状大小

```python
# Pytorch 写法
torch.empty(3, 5)

# Paddle 写法
paddle.empty([3, 5])
```

#### out：指定输出

```python
# Pytorch 写法
torch.empty((3, 5), out=y)

# Paddle 写法
paddle.assign(paddle.empty([3, 5]), y)
```

#### device: Tensor 的设备

```python
# Pytorch 写法
y = torch.empty((3, 5), device=torch.device('cpu'))

# Paddle 写法
y = paddle.empty([3, 5])
y.cpu()
```

#### requires_grad：是否求梯度

```python
# Pytorch 写法
y = torch.empty((3, 5), requires_grad=True)

# Paddle 写法
y = paddle.empty([3, 5])
y.stop_gradient = False
```

#### pin_memory：是否分配到固定内存上

```python
# Pytorch 写法
y = torch.empty((3, 5), pin_memory=True)

# Paddle 写法
y = paddle.empty([3, 5]).pin_memory()
```

