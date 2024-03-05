## [ 组合替代实现 ]torch.nn.init.kaiming_uniform_

### [torch.nn.init.kaiming_uniform_](https://pytorch.org/docs/stable/nn.init.html?highlight=kaiming_uniform_#torch.nn.init.kaiming_uniform_)

```python
torch.nn.init.kaiming_uniform_(tensor,
                        a=0,
                        mode='fan_in',
                        nonlinearity='leaky_relu')
```

### [paddle.nn.initializer.KaimingUniform](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/initializer/KaimingUniform_cn.html)

```python
paddle.nn.initializer.KaimingUniform(fan_in=None,
                            negative_slope=0.0,
                            nonlinearity='relu')
```

两者用法不同：torch 是 inplace 的用法，paddle 是类设置的，具体如下：

### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| tensor        | -          | n 维 tensor。Paddle 无此参数，因为是通过调用类的 __call__ 函数来进行 tensor 的初始化。    |
| a        | negative_slope     | 只适用于使用 leaky_relu 作为激活函数时的 negative_slope 参数。仅参数名不一致。    |
| nonlinearity     |  nonlinearity        |  非线性激活函数。参数默认值不一样，PyTorch 默认值为`leaky_relu`，Paddle 默认值为`relu`，Paddle 需保持与 PyTorch 一致。            |
| mode         | -        | "fan_in"（默认）或 "fan_out"。Paddle 无此参数，mode="fan_out"时，Paddle 无此参数，暂无转写方式。   |
| -          | fan_in        | 可训练的 Tensor 的 in_features 值。PyTorch 无此参数，Paddle 保持默认即可。               |

### 转写示例
```python
# PyTorch 写法
conv = torch.nn.Conv2d(4, 6, (3, 3))
torch.nn.init.kaiming_uniform_(conv.weight)

# Paddle 写法
conv = nn.Conv2D(in_channels=4, out_channels=6, kernel_size=(3,3))
init_kaimingUniform = paddle.nn.initializer.KaimingUniform(nonlinearity='leaky_relu')
init_kaimingUniform(conv.weight)
```
