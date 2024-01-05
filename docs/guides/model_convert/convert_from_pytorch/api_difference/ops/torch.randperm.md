## [torch 参数更多 ]torch.randperm

### [torch.randperm](https://pytorch.org/docs/stable/generated/torch.randperm.html?highlight=rand#torch.randperm)

```python
torch.randperm(n,
               *,
               generator=None,
               out=None,
               dtype=torch.int64,
               layout=torch.strided,
               device=None,
               requires_grad=False,
               pin_memory=False)
```

### [paddle.randperm](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/randperm_cn.html#randperm)

```python
paddle.randperm(n,
                dtype='int64',
                name=None)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                         |
| ------------- | ------------ | ------------------------------------------------------------ |
| n             | n            | 表示随机序列的上限。                                         |
| generator     | -            | 用于采样的伪随机数生成器， Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。 |
| out           | -            | 表示输出的 Tensor ， Paddle 无此参数，需要转写。         |
| dtype         | dtype        | 表示数据类型。                                               |
| layout        | -            | 表示布局方式， Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。 |
| device        | -            | 表示 Tensor 存放设备位置，Paddle 无此参数，需要转写。    |
| requires_grad | -            | 表示是否计算梯度， Paddle 无此参数，需要转写。           |
| pin_memeory   | -            | 表示是否使用锁页内存， Paddle 无此参数，需要转写。       |

### 转写示例

#### out：指定输出

```python
# PyTorch 写法
torch.randperm(10, out=y)

# Paddle 写法
paddle.assign(paddle.randperm(10), y)
```


#### requires_grad：是否求梯度

```python
# PyTorch 写法
x = torch.randperm(10, dtype=torch.float64,requires_grad=True)

# Paddle  写法
x = paddle.randperm(10)
x.stop_gradient = False
```

#### pin_memory：是否分配到固定内存上

```python
# PyTorch 写法
x = torch.randperm(10, pin_memory=True)

# Paddle 写法
x = paddle.randperm(10).pin_memory()
```

#### device: Tensor 的设备

```python
# PyTorch 写法
torch.randperm(10, device=torch.device('cpu'))

# Paddle 写法
y = paddle.randperm(10)
y.cpu()
```
