## [ torch 参数更多 ] torch.fft.fftfreq

### [torch.fft.fftfreq](https://pytorch.org/docs/stable/generated/torch.fft.fftfreq.html?highlight=fftfreq#torch.fft.fftfreq)

```python
torch.fft.fftfreq(n,
                d=1.0,
                *,
                out=None,
                dtype=None,
                layout=torch.strided,
                device=None,
                requires_grad=False)
```

### [paddle.fft.fftfreq](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/fft/fftfreq_cn.html)

```python
paddle.fft.fftfreq(n,
                    d=1.0,
                    dtype=None,
                    name=None)
```

其中，PyTorch 相比 Paddle 支持更多其他参数，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| n             | n            | 窗长度（傅里叶变换点数）。                        |
| d             | d            | 采样间隔，采样率的倒数，默认值为 1。                |
| out           | -            |输出的 Tensor，Paddle 无此参数，需要转写。              |
| dtype         | dtype        |返回 Tensor 的数据类型。                              |
| layout        |-             |表示布局方式，Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。|
| device        |-             |表示 Tensor 存放设备位置，Paddle 无此参数，需要转写。 |
| requires_grad |-             |表示是否不阻断梯度传导，Paddle 无此参数，需要转写。   |

### 转写示例
#### out：指定输出
```python
# PyTorch 写法
torch.fft.fftfreq(x, out=y)

# Paddle 写法
paddle.assign(paddle.fft.fftfreq(x),y)
```

#### requires_grad：是否需要求反向梯度，需要修改该 Tensor 的 stop_gradient 属性
```python
# PyTorch 写法
x = torch.fft.fftfreq(x, requires_grad=True)

# Paddle 写法
x = paddle.fft.fftfreq(x)
x.stop_gradient = False
```

#### device: Tensor 的设备
```python
# PyTorch 写法
torch.fft.fftfreq(x, device=torch.device('cpu'))

# Paddle 写法
y = paddle.fft.fftfreq(x)
y.cpu()
