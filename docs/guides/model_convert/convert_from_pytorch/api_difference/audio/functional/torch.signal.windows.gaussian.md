## [ torch 参数更多 ]torch.signal.windows.gaussian
### [torch.signal.windows.gaussian](https://pytorch.org/docs/stable/generated/torch.signal.windows.gaussian.html)

```python
torch.signal.windows.gaussian(M, *, std=1.0, sym=True, dtype=None, layout=torch.strided, device=None, requires_grad=False)
```

### [paddle.audio.functional.get_window](https://www.paddlepaddle.org.cn/documentation/docs/zh/2.6/api/paddle/audio/functional/get_window_cn.html#get-window)

```python
paddle.audio.functional.get_window(window, win_length, fftbins=True, dtype='float64')
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：
### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| - | window |  窗函数类型，Pytorch 无此参数，Paddle 需设置为 `gaussian`。 |
| M  | win_length            | 输入窗口的长度，也是采样点数。 |
| std  | -            | 高斯的标准差，Pytorch 默认值为 1.0, Paddle 无此参数，需要转写。 |
| sym        | fftbins       | 判断是否返回适用于过滤器设计的对称窗口。  |
| dtype        | dtype | 返回 Tensor 的数据类型。 |
| layout | -| 表示布局方式， Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。 |
| device | - | 表示 Tensor 存放设备位置，Paddle 无此参数，需要转写。 |
| requires_grad | - | 表示是否计算梯度， Paddle 无此参数，需要转写。 |

### 转写示例

#### window：窗函数类型
```python
# PyTorch 写法
torch.signal.windows.gaussian(10)

# Paddle 写法
std=1.0
paddle.audio.functional.get_window(('gaussian',std), 10)
```

#### std：高斯的标准差
```python
# PyTorch 写法
torch.signal.windows.gaussian(10, std=2.0)

# Paddle 写法
std=2.0
paddle.audio.functional.get_window(('gaussian',std), 10)
```

#### requires_grad：是否需要求反向梯度，需要修改该 Tensor 的 stop_gradient 属性
```python
# PyTorch 写法
torch.signal.windows.gaussian(10, requires_grad=True)

# Paddle 写法
std=1.0
x = paddle.audio.functional.get_window(('gaussian',std), 10)
x.stop_gradient = False
```

#### device: Tensor 的设备
```python
# PyTorch 写法
torch.signal.windows.gaussian(10, device=torch.device('cpu'))

# Paddle 写法
std=1.0
y = paddle.audio.functional.get_window(('gaussian',std), 10)
y.cpu()
```
