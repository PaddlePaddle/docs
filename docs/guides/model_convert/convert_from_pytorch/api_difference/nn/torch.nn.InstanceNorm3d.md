## [torch 参数更多]torch.nn.InstanceNorm3d

### [torch.nn.InstanceNorm3d](https://pytorch.org/docs/stable/generated/torch.nn.InstanceNorm3d.html#torch.nn.InstanceNorm3d)

```python
torch.nn.InstanceNorm3d(num_features, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False, device=None, dtype=None)
```

### [paddle.nn.InstanceNorm3D](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/InstanceNorm3D_cn.html)

```python
paddle.nn.InstanceNorm3D(num_features, epsilon=1e-05, momentum=0.9, weight_attr=None, bias_attr=None, data_format="NCDHW", name=None)
```

Pytorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch             | PaddlePaddle | 备注                                                            |
| ------------------- | ------------ | --------------------------------------------------------------- |
| num_features        | num_features | 输入 Tensor 的通道数量。                                        |
| eps                 | epsilon      | 为了数值稳定加在分母上的值，仅参数名不一致。                    |
| momentum            | momentum     | 此值用于计算 moving_mean 和 moving_var。                        |
| affine              | -            | 是否进行仿射变换，Paddle 无此参数，需要转写。               |
| track_running_stats | -            | 是否跟踪运行时的 mean 和 var， Paddle 无此参数。暂无转写方式。         |
| device              | -            | 表示 Tensor 存放设备位置，Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。  |
| dtype               | -            | 参数类型，Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。         |
| -                   | weight_attr  | 指定权重参数属性的对象，PyTorch 无此参数，Paddle 保持默认即可。 |
| -                   | bias_attr    | 指定偏置参数属性的对象，PyTorch 无此参数，Paddle 保持默认即可。 |
| -                   | data_format  | 指定输入数据格式，PyTorch 无此参数，Paddle 保持默认即可。       |

### 转写示例

#### affine：是否进行仿射变换

```python
# 当 PyTorch 的 affine 为`False`，表示 weight 和 bias 不进行更新，torch 写法
torch.nn.InstanceNorm3d(num_features, affine=False)

# paddle 写法
paddle.nn.InstanceNorm3d(num_features, weight_attr=False, bias_attr=False)

# 当 PyTorch 的 affine 为`True`，torch 写法
torch.nn.InstanceNorm3d(num_features, affine=True)

# paddle 写法
paddle.nn.InstanceNorm3d(num_features)
```
