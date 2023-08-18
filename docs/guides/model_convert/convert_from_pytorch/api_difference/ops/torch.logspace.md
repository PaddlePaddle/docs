## [torch 参数更多]torch.logspace

###  [torch.logspace](https://pytorch.org/docs/stable/generated/torch.logspace.html?highlight=logspace#torch.logspace)

```python
torch.logspace(start,
               end,
               steps,
               base=10.0,
               *,
               out=None,
               dtype=None,
               layout=torch.strided,
               device=None,
               requires_grad=False)
```

###  [paddle.logspace](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/logspace_cn.html)

```python
paddle.logspace(start,
                stop,
                num,
                base=10.0,
                dtype=None,
                name=None)
```

Pytorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                         |
| :------------ | :----------- | :----------------------------------------------------------- |
| start         | start        | 表示区间开始值以 base 为底的指数。                            |
| end           | stop         | 表示区间结束值以 base 为底的指数，仅参数名不一致。            |
| steps         | num          | 表示给定区间内需要划分的区间数，仅参数名不一致。             |
| base          | base         | 表示对数函数的底数。                                         |
| out           | -            | 表示输出的 Tensor，Paddle 无此参数，需要转写。           |
| dtype         | dtype        | 表示输出 Tensor 类型。                                       |
| layout        | -            | 表示布局方式，Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。 |
| device        | -            | 表示 Tensor 存放设备位置，Paddle 无此参数，需要转写。    |
| requires_grad | -            | 表示是否计算梯度，Paddle 无此参数，需要转写。            |

### 转写示例

#### out：指定输出

```python
# Pytorch 写法
torch.logspace(0, 10, 5, 2, out=y)

# Paddle 写法
paddle.assign(paddle.logspace(0, 10, 5, 2), y)
```

#### device: Tensor 的设备

```python
# Pytorch 写法
y = torch.logspace(0, 10, 5, 2, device=torch.device('cpu'))

# Paddle 写法
y = paddle.logspace(0, 10, 5, 2)
y.cpu()
```

#### requires_grad：是否求梯度

```python
# Pytorch 写法
y = torch.logspace(0, 10, 5, 2, requires_grad=True)

# Paddle 写法
y = paddle.logspace(0, 10, 5, 2)
y.stop_gradient = False
```
