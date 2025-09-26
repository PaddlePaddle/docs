## [ paddle 参数更多 ]torch.nn.LazyLinear
### [torch.nn.LazyLinear](https://pytorch.org/docs/stable/generated/torch.nn.LazyLinear.html)

```python
torch.nn.LazyLinear(out_features, bias=True, device=None, dtype=None)
```

### [paddle.nn.Linear](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/Linear_cn.html#linear)

```python
paddle.nn.Linear(in_features,
                 out_features,
                 weight_attr=None,
                 bias_attr=None,
                 name=None)
```

其中，Paddle 不支持 `in_features` 参数的延迟初始化，PyTorch 的 `bias` 与 Paddle 的 `bias_attr` 用法不一致，具体如下：
### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| -             | in_features   | 表示线性变换层输入单元的数目，PyTorch 无此参数，Paddle 需要根据实际输入 Tensor 的单元的数目进行设置。   |
| out_features  | out_features            | 表示线性变换层输出单元的数目。                             |
| bias          | -            | 是否在输出中添加可学习的 bias。                             |
| device        | -            | 指定 Tensor 的设备，Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。   |
| dtype         | -            | Tensor 的所需数据类型，Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。  |
| -             | weight_attr  | 指定权重参数的属性，PyTorch 无此参数，Paddle 保持默认即可。 |
| -             | bias_attr    | 指定偏置参数的属性, 当`bias_attr`设置为 bool 类型时与 PyTorch 的作用一致。 |

### 转写示例

#### in_channels: 输入通道数
在 PyTorch 中，使用 `LazyConvTranspose3d` 时可以不指定 `in_channels`，它会在第一次前向传播时根据输入 Tensor 的形状自动确定；而在 Paddle 中，创建 `Conv3DTranspose` 时必须明确指定 `in_channels` 参数，其值应与输入 Tensor 的通道数保持一致。
```python
# PyTorch 写法
linear = torch.nn.LazyLinear(out_features=10)
input = torch.randn(3, 5)  # 5 是输入单元数
output = linear(input)  # 此时 in_features 会根据输入 Tensor 的形状自动设置为 5

# Paddle 写法
linear = paddle.nn.Linear(in_features=5, out_features=10)  # 需要明确指定 in_features
input = paddle.randn([3, 5])  # 5 是输入单元数
output = linear(input)
```

#### bias: 是否在输出中添加可学习的 bias
```python
# PyTorch 写法
torch.nn.LazyLinear(out_features=4, bias=True)

# Paddle 写法
paddle.nn.Linear(in_features=2, out_features=4) # in_features 需要根据实际输入单元数进行设置
```
```python
# PyTorch 写法
torch.nn.LazyLinear(out_features=4, bias=False)

# Paddle 写法
paddle.nn.Linear(in_features=2, out_features=4, bias_attr=False) # in_features 需要根据实际输入单元数进行设置
```
