## [ 组合替代实现 ]torch.nn.init.orthogonal_

### [torch.nn.init.orthogonal_](https://pytorch.org/docs/stable/nn.init.html?highlight=orthogonal_#torch.nn.init.orthogonal_)

```python
torch.nn.init.orthogonal_(tensor,
                        gain=1)
```

### [paddle.nn.initializer.Orthogonal](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/initializer/Orthogonal_cn.html)

```python
paddle.nn.initializer.Orthogonal(gain=1.0,
                            name=None)
```

两者用法不同：torch 是 inplace 的用法，paddle 是类设置的。Pytorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| tensor        | -          | n 维 tensor。Paddle 无此参数，因为是通过调用类的 __call__ 函数来进行 tensor 的初始化。    |
| gain        | gain          |  参数初始化的增益系数。参数名和参数默认值均一致。    |

### 转写示例
```python
# Pytorch 写法
conv = torch.nn.Conv2d(4, 6, (3, 3))
torch.nn.init.orthogonal_(conv.weight, gain=2)

# Paddle 写法
conv = nn.Conv2D(in_channels=4, out_channels=6, kernel_size=(3,3))
init_orthogonal = paddle.nn.initializer.Orthogonal(gain=2)
init_orthogonal(conv.weight)
```
