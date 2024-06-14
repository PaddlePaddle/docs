## [ 组合替代实现 ]torch.nn.init.trunc_normal_

### [torch.nn.init.trunc_normal_](https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.trunc_normal_)

```python
torch.nn.init.trunc_normal_(tensor,
                            mean=0.0,
                            std=1.0,
                            a=-2.0,
                            b=2.0)
```

### [paddle.nn.initializer.TruncatedNormal](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/initializer/TruncatedNormal_cn.html)

```python
paddle.nn.initializer.TruncatedNormal(mean=0.0,
                                      std=1.0,
                                      a=-2.0,
                                      b=2.0,
                                      name=None)
```

两者用法不同：torch 是 inplace 的用法，paddle 是类设置的，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| tensor        | -          | n 维 tensor。Paddle 无此参数，因为是通过调用类的 __call__ 函数来进行 tensor 的初始化。    |
| mean          |  mean          | 正态分布的平均值。参数名和默认值一致。               |
| std           |  std         | 正态分布的标准差。参数名和默认值一致。               |
| a           |  a         | 截断正态分布的下界。参数名和默认值一致。               |
| b           |  b         | 截断正态分布的上界。参数名和默认值一致。               |

### 转写示例
```python
# PyTorch 写法
conv = torch.nn.Conv2d(4, 6, (3, 3))
torch.nn.init.trunc_normal_(conv.weight)

# Paddle 写法
conv = nn.Conv2D(in_channels=4, out_channels=6, kernel_size=(3,3))
init_trunc_normal = paddle.nn.initializer.TruncatedNormal()
init_trunc_normal(conv.weight)
```
