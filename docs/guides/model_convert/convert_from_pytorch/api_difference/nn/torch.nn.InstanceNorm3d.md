## [ 参数不一致 ]torch.nn.InstanceNorm3d
### [torch.nn.InstanceNorm3d](https://pytorch.org/docs/stable/generated/torch.nn.InstanceNorm3d.html?highlight=instancenorm3d#torch.nn.InstanceNorm3d)

```python
torch.nn.InstanceNorm3d(num_features,
                     eps=1e-05,
                     momentum=0.1,
                     affine=False,
                     track_running_stats=True,
                     device=None,
                     dtype=None)
```

### [paddle.nn.InstanceNorm3D](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/InstanceNorm3D_cn.html)

```python
paddle.nn.InstanceNorm3D(num_features,
                      epsilon=1e-05,
                      momentum=0.9,
                      weight_attr=None,
                      bias_attr=None,
                      data_format='NCDHW',
                      name=None)
```

两者功能一致, 部分参数用法一不致，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| num_features           | num_features      | 表示输入 Tensor 通道数。                                     |
| eps           | epsilon      | 为了数值稳定加在分母上的值，仅参数名不同。                                     |
| momentum           | momentum      | 表示归一化函数中的超参数， PyTorch 和 Paddle 公式实现细节不一致，两者正好是相反的，需要进行转写, Paddle 中 momentum 参数改写为 `1 - momentum`(Pytorch 中的参数)。                                     |
| -             | weight_attr  | 指定权重参数属性的对象。如果为 False, 则表示每个通道的伸缩固定为 1，不可改变。默认值为 None，表示使用默认的权重参数属性。转写时需注意当 PyTorch 的 affine 为`False`，表示 weight 不进行更新，paddle 对应参数设置为 paddle.ParamAttr(learning_rate=0.0) |
| -             | bias_attr    | 指定偏置参数属性的对象。如果为 False, 则表示每一个通道的偏移固定为 0，不可改变。默认值为 None，表示使用默认的偏置参数属性。转写时需注意当 PyTorch 的 affine 为`False`，表示 bias 不进行更新，Paddle 对应参数设置为 paddle.ParamAttr(learning_rate=0.0) |
| -             | data_format  | 指定输入数据格式， Pytorch 无此参数，Paddle 保持默认即可。 |
| affine        | -            | 是否进行反射变换， PaddlePaddle 无此参数，需要进行转写。         |
| track_running_stats | - | 无对应的参数，目前设置 track_running_stats 和 momentum 是无效的，之后的版本会修复此问题，可直接删除。|
| device | - | Tensor 的设备，一般对网络训练结果影响不大，可直接删除。         |
| dtype| - | 指定权重参数属性的对象，一般对网络训练结果影响不大，可直接删除。 |
### 转写示例
#### affine：weight 和 bias 不进行更新
```python
# 当 PyTorch 的 affine 为`False`，表示 weight 和 bias 不进行更新，torch 写法
torch.nn.InstanceNorm3d(num_features, eps=1e-05, momentum=0.1, affine=False)

# paddle 写法
paddle.nn.InstanceNorm3D(num_features=num_features, momentum=1 - 0.1, epsilon=1e-05, weight_attr=paddle.ParamAttr(learning_rate=0.0),
    bias_attr=paddle.ParamAttr(learning_rate=0.0))

```

#### affine：weight 和 bias 进行更新
```python
torch.nn.InstanceNorm3d(num_features, eps=1e-05, momentum=0.1, affine=True)

# paddle 写法
paddle.nn.InstanceNorm3D(num_features=num_features, momentum=1 - 0.1, epsilon=1e-05, weight_attr=None, bias_attr=None)
```
