## [torch 参数更多]torch.linspace

###  [torch.linspace](https://pytorch.org/docs/1.13/generated/torch.linspace.html?highlight=linspace#torch.linspace)

```python
torch.linspace(start,
               end,
               steps,
               *,
               out=None,
               dtype=None,
               layout=torch.strided,
               device=None,
               requires_grad=False)
```

###  [paddle.linspace](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/linspace_cn.html)

```python
paddle.linspace(start,
                stop,
                num,
                dtype=None,
                name=None)
```

其中 Pytorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                         |
| :------------ | :----------- | :----------------------------------------------------------- |
| start         | start        | 表示区间开始的变量。                                         |
| end           | stop         | 表示区间结束的变量，仅参数名不一致。                         |
| steps         | num          | 表示给定区间内需要划分的区间数，仅参数名不一致。             |
| out           | -            | 表示输出的 Tensor，Paddle 无此参数，需要进行转写。           |
| dtype         | dtype        | 表示输出 Tensor 的类型。                                     |
| layout        | -            | 表示布局方式，Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。 |
| device        | -            | 表示 Tensor 存放设备位置，Paddle 无此参数，需要进行转写。    |
| requires_grad | -            | 表示是否计算梯度，Paddle 无此参数，需要进行转写。            |

### 转写示例

#### out：指定输出

```python
# Pytorch 写法
torch.linspace(0, 10, 5, out=y)

# Paddle 写法
paddle.assign(paddle.linspace(0, 10, 5), y)
```

#### device: Tensor 的设备

```python
# Pytorch 写法
y = torch.linspace(0, 10, 5, device=torch.device('cpu'))

# Paddle 写法
y = paddle.linspace(0, 10, 5)
y.cpu()
```

#### requires_grad：是否求梯度

```python
# Pytorch 写法
y = torch.linspace(0, 10, 5, requires_grad=True)

# Paddle 写法
y = paddle.linspace(0, 10, 5)
y.stop_gradient = False
```

