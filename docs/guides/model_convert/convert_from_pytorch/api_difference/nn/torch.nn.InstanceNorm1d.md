## [ 参数不一致 ]torch.nn.InstanceNorm1d

### [torch.nn.InstanceNorm1d](https://pytorch.org/docs/stable/generated/torch.nn.InstanceNorm1d.html#torch.nn.InstanceNorm1d)

```python
torch.nn.InstanceNorm1d(num_features, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False, device=None, dtype=None)
```

### [paddle.nn.InstanceNorm1D](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/InstanceNorm1D_cn.html#instancenorm1d)
```python
paddle.nn.InstanceNorm1D(num_features, epsilon=1e-05, momentum=0.9, weight_attr=None, bias_attr=None, data_format="NCL", name=None)
```

Pytorch 相比 Paddle 支持更多其他参数，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| <font color='red'> num_features </font>   | <font color='red'> num_features </font>   | 指明输入的通道数量               |
| <font color='red'> eps  </font>         |    <font color='red'> epsilon  </font>         | 为了数值稳定加在分母上的值             |
| <font color='red'> momentum </font>             | <font color='red'> momentum </font>  | 此值用于计算 moving_mean 和 moving_var, 值的大小 Paddle = 1 - Pytorch，需要转写。               |
| <font color='red'> affine </font>             | -  | 是否使用可学习的仿射参数，Paddle 无此参数，需要转写。可通过 weight_attr 和 bias_attr 控制。             |
| <font color='red'> track_running_stats </font>           |  -            | 是否跟踪运行时的 mean 和 var， Paddle 无此参数，暂无转写方式。  |
| <font color='red'> dtype </font>           |  -            | 输出数据类型， Paddle 无此参数, 需要转写。Paddle 应使用 astype 对计算结果做类型转换。  |
| -           |  <font color='red'> weight_attr </font>            | 可学习参数——权重的属性，默认为 None，表示使用默认可学习参数。 Pytorch 无此参数。 |
| -           |  <font color='red'> bias_attr </font>            | 可学习参数——偏差的属性，默认为 None，表示使用默认可学习参数。 Pytorch 无此参数。 |
| -           |  <font color='red'> data_format </font>            | 指定输入数据格式。 Pytorch 无此参数。 |


### 转写示例
#### affine：是否使用可学习的仿射参数
```python
# Pytorch 写法
IN = torch.nn.InstanceNorm1d(num_features, eps=1e-05, momentum=0.1, affine=False)

# Paddle 写法
IN = paddle.nn.InstanceNorm1D(num_features, epsilon=1e-05, momentum=0.9, weight_attr=False, bias_attr=False)
```

#### dtype：输出数据类型
```python
# Pytorch 写法
IN = torch.nn.InstanceNorm1d(num_features, eps=1e-05, momentum=0.1, affine=False， dtype=torch.float32)
y = IN(x)

# Paddle 写法
IN = paddle.nn.InstanceNorm1D(num_features, epsilon=1e-05, momentum=0.9, weight_attr=False, bias_attr=False)
y = IN(x).astype(paddle.float32)
```
