## [ 组合替代实现 ]torch.nn.init.uniform_.md

### [torch.nn.init.uniform_](https://pytorch.org/docs/stable/nn.init.html?highlight=uniform_#torch.nn.init.uniform_)

```python
torch.nn.init.uniform_(tensor,
                        a=0.0,
                        b=1.0)
```

### [paddle.nn.initializer.Uniform](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/initializer/Uniform_cn.html)

```python
paddle.nn.initializer.Uniform(low=-1.0,
                            high=1.0,
                            name=None)
```

两者用法不同：torch 是 inplace 的用法，paddle 是类设置的。Pytorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| tensor        | -          | n 维 tensor。Paddle 无此参数，因为是通过调用类的 __call__ 函数来进行 tensor 的初始化。    |
| a           |  low          | 均匀分布的下界，参数默认值不一致, Pytorch 默认为`0.0`，Paddle 为`-1.0`，Paddle 需保持与 Pytorch 一致。               |
| b           |  high         | 均匀分布的上界，仅参数名不一致。               |

### 转写示例
```python
# Pytorch 写法
conv = torch.nn.Conv2d(4, 6, (3, 3))
torch.nn.init.uniform_(conv.weight)

# Paddle 写法
conv = nn.Conv2D(in_channels=4, out_channels=6, kernel_size=(3,3))
init_uniform = paddle.nn.initializer.Uniform(low=0.0)
init_uniform(conv.weight)
```
