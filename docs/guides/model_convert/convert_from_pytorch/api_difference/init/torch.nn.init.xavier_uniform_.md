## [ 组合替代实现 ]torch.nn.init.xavier_uniform_.md

### [torch.nn.init.xavier_uniform_](https://pytorch.org/docs/stable/nn.init.html?highlight=xavier_uniform_#torch.nn.init.xavier_uniform_)

```python
torch.nn.init.xavier_uniform_(tensor,
                        gain=1.0)
```

### [paddle.nn.initializer.XavierUniform](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/initializer/XavierUniform_cn.html)

```python
paddle.nn.initializer.XavierUniform(fan_in=None,
                            fan_out=None,
                            name=None)
```

两者用法不同：torch 是 inplace 的用法，paddle 是类设置的，具体如下：

### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| tensor        | -          | n 维 tensor。Paddle 无此参数，因为是通过调用类的 __call__ 函数来进行 tensor 的初始化。    |
| gain        | -          |  缩放因子。Paddle 无此参数，暂无转写方式。    |
| -          |  fan_in          | 用于泽维尔初始化的 fan_in。Pytorch 无此参数，Paddle 保持默认即可。        |
| -          |  fan_out         | 用于泽维尔初始化的 fan_out。Pytorch 无此参数，Paddle 保持默认即可。       |

### 转写示例
```python
# Pytorch 写法
conv = torch.nn.Conv2d(4, 6, (3, 3))
torch.nn.init.xavier_uniform_(conv.weight)

# Paddle 写法
conv = nn.Conv2D(in_channels=4, out_channels=6, kernel_size=(3,3))
init_xaiverUniform = paddle.nn.initializer.XavierUniform()
init_xaiverUniform(conv.weight)
```
