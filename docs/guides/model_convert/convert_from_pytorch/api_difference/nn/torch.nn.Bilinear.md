## [torch 参数更多]torch.nn.Bilinear

### [torch.nn.Bilinear](https://pytorch.org/docs/stable/generated/torch.nn.Bilinear.html#torch.nn.Bilinear)

```python
torch.nn.Bilinear(in1_features, in2_features, out_features, bias=True, device=None, dtype=None)
```

### [paddle.nn.Bilinear](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/Bilinear_cn.html)

```python
paddle.nn.Bilinear(in1_features, in2_features, out_features, weight_attr=None, bias_attr=None, name=None)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch      | PaddlePaddle | 备注                                                            |
| ------------ | ------------ | --------------------------------------------------------------- |
| in1_features | in1_features | 每个 x1 元素的维度。                                            |
| in2_features | in2_features | 每个 x2 元素的维度。                                            |
| out_features | out_features | 输出张量的维度。                                                |
| bias         | bias_attr    | 指定偏置参数属性的对象，Paddle 支持更多功能，同时支持 bool 用法。   |
| device       | -            | Tensor 的设备，Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。 |
| dtype        | -            | Tensor 的数据类型，Paddle 无此参数，一般对网络训练结果影响不大，可直接删除 |
| -            | weight_attr  | 指定权重参数属性的对象，PyTorch 无此参数，Paddle 保持默认即可。 |

### 转写示例

#### device：Tensor 的设备

```python
# PyTorch 写法
m = torch.nn.Bilinear(in1_features, in2_features, out_features，device=torch.device('cpu'))
y = m(x)

# Paddle 写法
m = paddle.nn.Bilinear(in1_features, in2_features, out_features)
y = m(x).cpu()
```

#### dtype：Tensor 的数据类型

```python
# PyTorch 写法
m = torch.nn.Bilinear(in1_features, in2_features, out_features，dtype=torch.float32)
y = m(x)

# Paddle 写法
m = paddle.nn.Bilinear(in1_features, in2_features, out_features)
y = m(x).astype(paddle.float32)
```
