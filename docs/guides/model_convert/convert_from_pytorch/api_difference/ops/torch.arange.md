## [torch 参数更多]torch.arange

###  [torch.arange](https://pytorch.org/docs/1.13/generated/torch.arange.html?highlight=arange#torch.arange)

```python
torch.arange(start=0,
             end,
             step=1,
             *,
             out=None,
             dtype=None,
             layout=torch.strided,
             device=None,
             requires_grad=False)
```

###  [paddle.arange](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/arange_cn.html)

```python
paddle.arange(start=0,
              end=None,
              step=1,
              dtype=None,
              name=None)
```

其中 Pytorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                         |
| :------------ | :----------- | :----------------------------------------------------------- |
| start         | start        | 表示区间起点（且区间包括此值）。                             |
| end           | end          | 表示区间终点（且通常区间不包括此值）。                       |
| step          | step         | 表示均匀分割的步长。                                         |
| out           | -            | 表示输出的 Tensor，Paddle 无此参数，需要进行转写。           |
| dtype         | dtype        | 表示输出 Tensor 类型。                                       |
| layout        | -            | 表示布局方式，Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。 |
| device        | -            | 表示 Tensor 存放设备位置，Paddle 无此参数，需要进行转写。    |
| requires_grad | -            | 表示是否计算梯度，Paddle 无此参数，需要进行转写。            |

### 转写示例

#### out：指定输出

```python
# Pytorch 写法
torch.arange(5, out=y)

# Paddle 写法
paddle.assign(paddle.arange(5), y)
```

#### device: Tensor 的设备

```python
# Pytorch 写法
y = torch.arange(5, device=torch.device('cpu'))

# Paddle 写法
y = paddle.arange(5)
y.cpu()
```

#### requires_grad：是否求梯度

```python
# Pytorch 写法
y = torch.arange(5, requires_grad=True)

# Paddle 写法
y = paddle.arange(5)
y.stop_gradient = False
```
