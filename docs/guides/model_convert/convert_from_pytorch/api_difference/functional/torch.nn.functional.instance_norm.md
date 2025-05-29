## [ 输入参数用法不一致 ]torch.nn.functional.instance_norm

### [torch.nn.functional.instance_norm](https://pytorch.org/docs/stable/generated/torch.nn.functional.instance_norm.html#torch.nn.functional.instance_norm)

```python
torch.nn.functional.instance_norm(input, running_mean=None, running_var=None, weight=None, bias=None, use_input_stats=True, momentum=0.1, eps=1e-05)
```

### [paddle.nn.functional.instance_norm](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/functional/instance_norm_cn.html#instance-norm)
```python
paddle.nn.functional.instance_norm(x, running_mean=None, running_var=None, weight=None, bias=None, training=False, eps=1e-05, momentum=0.9, use_input_stats=True, data_format='NCHW', name=None)
```

其中 PyTorch 与 Paddle 参数不一致，具体如下：
### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
|  input  |  x  | 表示输入的 Tensor ，仅参数名不一致。  |
|  running_mean    |  running_mean    | 均值的 Tensor               |
|  running_var    |  running_var    | 方差的 Tensor               |
|  weight   | weight    | 权重的 Tensor               |
|  bias    |  bias    | 偏置的 Tensor               |
|  eps           |     eps           | 为了数值稳定加在分母上的值             |
|  momentum              |  momentum   | 此值用于计算 moving_mean 和 moving_var, 值的大小 Paddle = 1 - PyTorch，需要转写               |
| -           |   training            | 是否可训练。 PyTorch 无此参数。保持默认即可。 |
| -           |   data_format             | 指定输入数据格式。 PyTorch 无此参数。保持默认即可。 |


### 转写示例
#### momentum：此值用于计算 moving_mean 和 moving_var
```python
# PyTorch 写法
torch.nn.functional.instance_norm(input=input, running_mean=running_mean, running_var=running_var, momentum=0.1)

# Paddle 写法
paddle.nn.functional.instance_norm(x=input, running_mean=running_mean, running_var=running_var, momentum=0.9)
```
