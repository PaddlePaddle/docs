## [ paddle 参数更多 ]torch.nn.LazyBatchNorm3d
### [torch.nn.LazyBatchNorm3d](https://pytorch.org/docs/stable/generated/torch.nn.LazyBatchNorm3d.html)

```python
torch.nn.LazyBatchNorm3d(eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None)
```

### [paddle.nn.BatchNorm3D](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/BatchNorm3D_cn.html#batchnorm3d)

```python
paddle.nn.BatchNorm3D(num_features,
                      momentum=0.9,
                      epsilon=1e-05,
                      weight_attr=None,
                      bias_attr=None,
                      data_format='NCDHW',
                      use_global_stats=True,
                      name=None)
```

其中，Paddle 不支持 `num_features` 参数的延迟初始化，部分参数名不同，具体如下：
### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| -             | num_features   | 表示输入 Tensor 通道数，PyTorch 无此参数，Paddle 需要根据实际输入 Tensor 的通道数进行设置。                             |
| eps           | epsilon      | 为了数值稳定加在分母上的值，仅参数名不一致。                                                                                                      |
| momentum      | momentum      | 表示归一化函数中的超参数, PyTorch 和 Paddle 公式实现细节不一致，两者正好是相反的，需要转写。                                                                     |
| -             | weight_attr  | 指定权重参数属性的对象。如果为 False, 则表示每个通道的伸缩固定为 1，不可改变。默认值为 None，表示使用默认的权重参数属性。                                                        |
| -             | bias_attr    | 指定偏置参数属性的对象。如果为 False, 则表示每一个通道的偏移固定为 0，不可改变。默认值为 None，表示使用默认的偏置参数属性。                                                       |
| -             | data_format  | 指定输入数据格式，PyTorch 无此参数，Paddle 保持默认即可。                                                                                        |
| affine        | -                | 是否进行反射变换， Paddle 无此参数，需要转写。                                                                                                 |
| track_running_stats | use_global_stats | 指示是否使用全局均值和方差，PyTorch 设置为 True，Paddle 需设置为 False；PyTorch 设置为 None，Paddle 需设置为 True；PyTorch 设置为 False，Paddle 需设置为 True，需要转写。 |
| device        | -            | 指定 Tensor 的设备，一般对网络训练结果影响不大，可直接删除。   |
| dtype         | -            | 指定权重参数属性的对象，一般对网络训练结果影响不大，可直接删除。 |

### 转写示例

#### num_features: 输入通道数
在 PyTorch 中，使用 `LazyBatchNorm3d` 时可以不指定 `num_features`，它会在第一次前向传播时根据输入 Tensor 的形状自动确定；而在 Paddle 中，创建 `BatchNorm3D` 时必须明确指定 `num_features` 参数，其值应与输入 Tensor 的通道数保持一致。
```python
# PyTorch 写法
bn = torch.nn.LazyBatchNorm3d()
input = torch.randn(3, 5, 32, 32, 32)  # 5 是输入通道数
output = bn(input)  # 此时 num_features 会根据输入 Tensor 的形状自动设置为 5

# Paddle 写法
bn = paddle.nn.BatchNorm3D(num_features=5)  # 需要明确指定 num_features
input = paddle.randn([3, 5, 32, 32, 32])  # 5 是输入通道数
output = bn(input)
```

#### affine：是否进行反射变换
```python
affine=False 时，表示不更新：

# PyTorch 写法
m = torch.nn.LazyBatchNorm3d(affine=False)

# Paddle 写法
m = paddle.nn.BatchNorm3D(num_features=24, weight_attr=False, bias_attr=False) # num_features 需要根据实际输入通道数进行设置

affine=True 时，表示更新：

# PyTorch 写法
m = torch.nn.LazyBatchNorm3d()

# Paddle 写法
m = paddle.nn.BatchNorm3D(num_features=24) # num_features 需要根据实际输入通道数进行设置
```

#### momentum：
```python
# PyTorch 写法
m = torch.nn.LazyBatchNorm3d(momentum=0.2)

# Paddle 写法
m = paddle.nn.BatchNorm3D(num_features=24, momentum=0.8) # num_features 需要根据实际输入通道数进行设置
```

#### track_running_stats：指示是否使用全局均值和方差

```python
track_running_stats=None 时:
# PyTorch 写法
m = torch.nn.LazyBatchNorm3d(track_running_stats=None)

# Paddle 写法
m = paddle.nn.BatchNorm3D(num_features=24, use_global_stats=True) # num_features 需要根据实际输入通道数进行设置

track_running_stats=True 时:
# PyTorch 写法
m = torch.nn.LazyBatchNorm3d(track_running_stats=True)

# Paddle 写法
m = paddle.nn.BatchNorm3D(num_features=24, use_global_stats=False) # num_features 需要根据实际输入通道数进行设置

track_running_stats=False 时:
# PyTorch 写法
m = torch.nn.LazyBatchNorm3d(track_running_stats=False)

# Paddle 写法
m = paddle.nn.BatchNorm3D(num_features=24, use_global_stats=True) # num_features 需要根据实际输入通道数进行设置
```
