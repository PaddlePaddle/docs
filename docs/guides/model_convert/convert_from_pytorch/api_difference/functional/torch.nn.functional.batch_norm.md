## [ 参数不一致 ]torch.nn.functional.batch_norm

### [torch.nn.functional.batch_norm](https://pytorch.org/docs/stable/generated/torch.nn.functional.batch_norm.html#torch.nn.functional.batch_norm)

```python
torch.nn.functional.batch_norm(input, running_mean, running_var, weight=None, bias=None, training=False, momentum=0.1, eps=1e-05)
```

### [paddle.nn.functional.batch_norm](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/functional/batch_norm_cn.html#batch-norm)
```python
paddle.nn.functional.batch_norm(x, running_mean, running_var, weight, bias, training=False, momentum=0.9, epsilon=1e-05, data_format='NCHW', name=None)
```

其中 Pytorch 与 Paddle 参数不一致，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| <font color='red'> input </font> | <font color='red'> x </font> | 表示输入的 Tensor ，仅参数名不一致。  |
| <font color='red'> running_mean </font>  | <font color='red'> running_mean </font> | 均值的 Tensor。 |
| <font color='red'> running_var </font>   | <font color='red'> running_var </font>  | 方差的 Tensor。 |
| <font color='red'> weight </font>   | <font color='red'> weight </font>   | 权重的 Tensor。          |
| <font color='red'> bias </font>   | <font color='red'> bias </font>   | 偏置的 Tensor。              |
| <font color='red'> eps  </font>         |    <font color='red'> epsilon  </font>         | 为了数值稳定加在分母上的值。       |
| <font color='red'> momentum </font>             | <font color='red'> momentum </font>  | 此值用于计算 moving_mean 和 moving_var, 值的大小 Paddle = 1 - Pytorch，需要转写。      |
| <font color='red'> training </font>           |  <font color='red'> training </font>            | 是否可训练。 |
| -  |  <font color='red'> data_format </font> | 指定输入数据格式，Pytorch 无此参数，Paddle 保持默认即可。 |


### 转写示例
#### momentum：此值用于计算 moving_mean 和 moving_var
```python
# Pytorch 写法
torch.nn.functional.batch_norm(input=input, running_mean=running_mean, running_var=running_var, momentum=0.1)

# Paddle 写法
paddle.nn.functional.batch_norm(x=input, running_mean=running_mean, running_var=running_var, momentum=0.9)
```
