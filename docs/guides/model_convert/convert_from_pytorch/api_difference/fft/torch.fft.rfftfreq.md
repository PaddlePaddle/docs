## [torch 参数更多] torch.fft.rfftfreq

### [torch.fft.rfftfreq](https://pytorch.org/docs/1.13/generated/torch.fft.rfftfreq.html?highlight=rfftfreq#torch.fft.rfftfreq)

```python
torch.fft.rfftfreq(n,
                d=1.0,
                *,
                out=None,
                dtype=None,
                layout=torch.strided,
                device=None,
                requires_grad=False)
```

### [paddle.fft.rfftfreq](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/fft/rfftfreq_cn.html)

```python
paddle.fft.rfftfreq(n,
                    d=1.0,
                    dtype=None,
                    name=None)
```

两者功能一致但参数不一致，部分参数名不同，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| n             | n                | 窗长度（傅里叶变换点数），参数名相同。                        |
| d             | d            | 采样间隔，采样率的倒数，默认值为 1。 参数名相同。         |
| out            | -            |输出的 Tensor,Paddle 无此参数，需要进行转写。              |
| dtype          | dtype      | 返回 Tensor 的数据类型。参数名相同。|
|layout         |-            |表示布局方式，Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。|
|device         |-              | 表示 Tensor 存放设备位置，Paddle 无此参数，需要进行转写。         |
|requires_grad  |-             |  表示是否不阻断梯度传导，Paddle 无此参数，需要进行转写。    |

### 转写示例
#### out：指定输出
```python
# Pytorch 写法
torch.fft.rfftfreq(5, out=y)

# Paddle 写法
paddle.assign(paddle.fft.rfftfreq(np.array([3, 1, 2, 2, 3], dtype=float).size, d=0.5),y)
```

#### requires_grad：是否需要求反向梯度，需要修改该 Tensor 的 stop_gradient 属性
```python
# Pytorch 写法
x = torch.fft.rfftfreq(5, requires_grad=True)
# Paddle 写法
x = paddle.fft.rfftfreq(np.array([3, 1, 2, 2, 3], dtype=float).size, d=0.5)
x.stop_gradient = False
```

#### device: Tensor 的设备
```python
# Pytorch 写法
torch.fft.rfftfreq(5, device=torch.device('cpu'))
# Paddle 写法
y = paddle.fft.rfftfreq(np.array([3, 1, 2, 2, 3], dtype=float).size, d=0.5)
y.cpu()
