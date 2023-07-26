## [ 组合替代实现 ]torch.nn.init.dirac_.md

### [torch.nn.init.dirac_](https://pytorch.org/docs/stable/nn.init.html?highlight=dirac_#torch.nn.init.dirac_)

```python
torch.nn.init.dirac_(tensor,
                    groups=1)
```

### [paddle.nn.initializer.Dirac](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/initializer/Dirac_cn.html)

```python
paddle.nn.initializer.Dirac(groups=1,
                            name=None)
```

两者用法不同：torch 是 inplace 的用法，paddle 是类设置的。Pytorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| tensor        | -          | n 维 tensor。Paddle 无此参数，因为是通过调用类的 __call__ 函数来进行 tensor 的初始化。    |
| groups        |  groups       | 将参数在 0 维上进行等分为 groups 份，每一份执行相同的初始化。参数名和默认值一致。               |

### 转写示例
```python
# Pytorch 写法
conv = torch.nn.Conv2d(4, 6, (3, 3))
torch.nn.init.dirac_(conv.weight)

# Paddle 写法
conv = nn.Conv2D(in_channels=4, out_channels=6, kernel_size=(3,3))
init_dirac = paddle.nn.initializer.Dirac()
init_dirac(conv.weight)
```
