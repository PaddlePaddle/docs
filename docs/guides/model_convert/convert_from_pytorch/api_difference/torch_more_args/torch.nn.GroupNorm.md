## [ torch 参数更多 ]torch.nn.GroupNorm
### [torch.nn.GroupNorm](https://pytorch.org/docs/stable/generated/torch.nn.GroupNorm.html?highlight=groupnorm#torch.nn.GroupNorm)

```python
torch.nn.GroupNorm(num_groups,
                   num_channels,
                   eps=1e-05,
                   affine=True,
                   device=True,
                   dtype=None)
```

### [paddle.nn.GroupNorm](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/GroupNorm_cn.html#groupnorm)

```python
paddle.nn.GroupNorm(num_groups,
                    num_channels,
                    epsilon=1e-05,
                    weight_attr=None,
                    bias_attr=None,
                    data_format='NCHW',
                    name=None)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| eps           | epsilon      | 为了数值稳定加在分母上的值。                                     |
| affine        | -            | 是否进行仿射变换，Paddle 无此参数，需要转写。         |
| device        | -            | 设备类型，Paddle 无此参数。         |
| dtype         | -            | 参数类型，Paddle 无此参数。         |
| -             | weight_attr  | 指定权重参数属性的对象。如果为 False, 则表示每个通道的伸缩固定为 1，不可改变。默认值为 None，表示使用默认的权重参数属性。 |
| -             | bias_attr    | 指定偏置参数属性的对象。如果为 False, 则表示每一个通道的偏移固定为 0，不可改变。默认值为 None，表示使用默认的偏置参数属性。 |
| -             | data_format  | 指定输入数据格式，PyTorch 无此参数，Paddle 保持默认即可。 |

### 转写示例
#### affine：是否进行仿射变换
```python
# 当 PyTorch 的 affine 为`False`，表示 weight 和 bias 不进行更新，torch 写法
torch.nn.GroupNorm(num_groups=5, num_channels=50, eps=1e-05, affine=False)

# paddle 写法
paddle.nn.GroupNorm(num_groups=5, num_channels=50, epsilon=1e-05, weight_attr=False, bias_attr=False)

# 当 PyTorch 的 affine 为`True`，torch 写法
torch.nn.GroupNorm(num_groups=5, num_channels=50, eps=1e-05, affine=True)

# paddle 写法
paddle.nn.GroupNorm(num_groups=5, num_channels=50, epsilon=1e-05)
```
