## torch.nn.GroupNorm
### [torch.nn.GroupNorm](https://pytorch.org/docs/stable/generated/torch.nn.GroupNorm.html?highlight=groupnorm#torch.nn.GroupNorm)

```python
torch.nn.BatchNorm3d(num_groups,
                     num_channels,
                     eps=1e-05,
                     affine=True,
                     device=True,
                     dtype=None)
```

### [paddle.nn.GroupNorm](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/GroupNorm_cn.html#groupnorm)

```python
paddle.nn.GroupNorm(num_groups,
                    num_channels,
                    epsilon=1e-05,
                    weight_attr=None,
                    bias_attr=None,
                    data_format='NCHW',
                    name=None)
```

两者功能一致但参数不一致，部分参数名不同，具体如下：
### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| eps           | epsilon      | 为了数值稳定加在分母上的值。                                     |
| -             | weight_attr  | 指定权重参数属性的对象。如果为 False, 则表示每个通道的伸缩固定为 1，不可改变。默认值为 None，表示使用默认的权重参数属性。 |
| -             | bias_attr    | 指定偏置参数属性的对象。如果为 False, 则表示每一个通道的偏移固定为 0，不可改变。默认值为 None，表示使用默认的偏置参数属性。 |
| -             | data_format  | 指定输入数据格式，Pytorch 无此参数，Paddle 保持默认即可。 |
| affine        | -            | 是否进行反射变换，PaddlePaddle 无此参数。         |
| device        | data_format  | 指定输入数据格式，Pytorch 无此参数，Paddle 保持默认即可。 |
| dtype         | -            | 是否进行反射变换，PaddlePaddle 无此参数。         |

### 转写示例
#### affine：是否进行反射变换
```python
# 当 PyTorch 的反射变换设置为`False`，表示 weight 和 bias 不进行更新，Paddle 可用代码组合实现该 API
class GroupNorm(paddle.nn.GroupNorm):
    def __init__(self,
                 num_groups,
                 num_channels,
                 eps=1e-05,
                 affine=True):
        weight_attr = None
        bias_attr = None
        if not affine:
            weight_attr = paddle.ParamAttr(learning_rate=0.0)
            bias_attr = paddle.ParamAttr(learning_rate=0.0)
        super().__init__(
            num_groups=num_groups,
            num_channels=num_channels,
            epsilon=1e-05,
            weight_attr=weight_attr,
            bias_attr=bias_attr,
            data_format='NCHW')
```
