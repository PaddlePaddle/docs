## [ 组合替代实现 ]torch.nn.init.constant_

### [torch.nn.init.constant_](https://pytorch.org/docs/stable/nn.init.html?highlight=constant_#torch.nn.init.constant_)

```python
torch.nn.init.constant_(tensor,
                        val)
```

### [paddle.nn.initializer.Constant](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/initializer/Constant_cn.html)

```python
paddle.nn.initializer.Constant(value=0.0)
```

两者用法不同：torch 是 inplace 的用法，paddle 是类设置的。PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| tensor        | -          | n 维 tensor。Paddle 无此参数，因为是通过调用类的 __call__ 函数来进行 tensor 的初始化。    |
| val          |  value          | 用于初始化输入变量的值。PyTorch 无默认值，Paddle 默认值为`0.0`。               |

### 转写示例
```python
# PyTorch 写法
conv = torch.nn.Conv2d(4, 6, (3, 3))
torch.nn.init.constant_(conv.weight, val=1.0)

# Paddle 写法
conv = nn.Conv2D(in_channels=4, out_channels=6, kernel_size=(3,3))
init_constant = paddle.nn.initializer.Constant(value=1.0)
init_constant(conv.weight)
```
