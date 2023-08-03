## [ 参数不一致 ]torch.nn.InstanceNorm2d

### [torch.nn.InstanceNorm2d](https://pytorch.org/docs/stable/generated/torch.nn.InstanceNorm2d.html#torch.nn.InstanceNorm2d)

```python
torch.nn.InstanceNorm2d(num_features, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False, device=None, dtype=None)
```

### [paddle.nn.InstanceNorm2D](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/InstanceNorm2D_cn.html#instancenorm2d)
```python
paddle.nn.InstanceNorm2D(num_features, epsilon=1e-05, momentum=0.9, weight_attr=None, bias_attr=None, data_format="NCL", name=None)
```

Pytorch 相比 Paddle 支持更多其他参数，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| <font color='red'> num_features </font>   | <font color='red'> num_features </font>   | 指明输入的通道数量。               |
| <font color='red'> eps  </font>         |    <font color='red'> epsilon  </font>         | 为了数值稳定加在分母上的值。             |
| <font color='red'> momentum </font>             | <font color='red'> momentum </font>  | 此值用于计算 moving_mean 和 moving_var，值的大小 Paddle = 1 - Pytorch，需要转写。               |
| <font color='red'> affine </font>             | -  | 是否使用可学习的仿射参数，Paddle 无此参数。可通过 weight_attr 和 bias_attr 控制。              |
| <font color='red'> track_running_stats </font>           |  -            | 是否跟踪运行时的 mean 和 var， Paddle 无此参数。暂无转写方式。  |
| <font color='red'> dtype </font>           |  -            | 输出数据类型， Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。 |
| -           |  <font color='red'> weight_attr </font>            | 可学习参数——权重的属性，默认为 None，表示使用默认可学习参数。 Pytorch 无此参数。 |
| -           |  <font color='red'> bias_attr </font>            | 可学习参数——偏差的属性，默认为 None，表示使用默认可学习参数。 Pytorch 无此参数。 |
| -           |  <font color='red'> data_format </font>            | 指定输入数据格式。 Pytorch 无此参数。 |


### 转写示例
#### affine：是否使用可学习的仿射参数
```python
# Pytorch 写法
IN = torch.nn.InstanceNorm2d(num_features, eps=1e-05, momentum=0.1, affine=False)

# Paddle 写法
IN = paddle.nn.InstanceNorm2D(num_features, epsilon=1e-05, momentum=0.9, weight_attr=False, bias_attr=False)
```
