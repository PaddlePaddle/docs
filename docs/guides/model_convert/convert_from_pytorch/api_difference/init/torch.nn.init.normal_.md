## [ 组合替代实现 ]torch.nn.init.normal_.md

### [torch.nn.init.normal_](https://pytorch.org/docs/stable/nn.init.html?highlight=normal_#torch.nn.init.normal_)

```python
torch.nn.init.normal_(tensor,
                        mean=0.0,
                        std=1.0)
```

### [paddle.nn.initializer.Normal](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/initializer/Normal_cn.html)

```python
paddle.nn.initializer.Normal(mean=0.0,
                            std=1.0,
                            name=None)
```

两者用法不同：torch 是 inplace 的用法，paddle 是类设置的。Pytorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| tensor        | -          | n 维 tensor。Paddle 无此参数，因为是通过调用类的 __call__ 函数来进行 tensor 的初始化。    |
| mean          |  mean          | 正态分布的平均值。参数名和默认值一致。               |
| std           |  std         | 正态分布的标准差。参数名和默认值一致。               |

### 转写示例
```python
# Pytorch 写法
conv = torch.nn.Conv2d(4, 6, (3, 3))
torch.nn.init.normal_(conv.weight)

# Paddle 写法
conv = nn.Conv2D(in_channels=4, out_channels=6, kernel_size=(3,3))
init_normal = paddle.nn.initializer.Normal()
init_normal(conv.weight)
```
