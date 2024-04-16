## [ torch 参数更多 ]torch.nn.modules.batchnorm._BatchNorm

### [torch.nn.modules.batchnorm.\_BatchNorm](https://pytorch.org/docs/stable/_modules/torch/nn/modules/batchnorm.html)

```python
torch.nn.modules.batchnorm._BatchNorm(num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None)
```

### [paddle.nn.layer.norm.\_BatchNormBase](https://github.com/PaddlePaddle/Paddle/blob/b51d50bc9ee9eaa5cefa18507195b239e4513194/python/paddle/nn/layer/norm.py#L701)

```python
paddle.nn.layer.norm._BatchNormBase(num_features, momentum=0.9, epsilon=1e-05, weight_attr=None, bias_attr=None, data_format='NCHW', use_global_stats=None, name=None)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch             | PaddlePaddle                 | 备注       |
| ------------------- | ---------------------------- | --------- |
| num_features        | num_features                 | 特征数量。 |
| eps                 | epsilon                      | 一个极小正数，仅参数名不一致。 |
| momentum            | momentum                     | 动量因子，参数默认值不一致。 |
| affine              | -                            | 是否进行仿射变换，Paddle 无此参数，需要转写。
| track_running_stats | -                            | 是否跟踪运行时的 mean 和 var， Paddle 无此参数。暂无转写方式。         |
| device              | -                            | 表示 Tensor 存放设备位置，Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。  |
| dtype               | -                            | 参数类型，Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。         |
| -                   | weight_attr                  | 指定权重参数属性的对象，PyTorch 无此参数，Paddle 保持默认即可。 |
| -                   | bias_attr                    | 指定偏置参数属性的对象，PyTorch 无此参数，Paddle 保持默认即可。 |
| -                   | data_format                  | 指定输入数据格式，PyTorch 无此参数，Paddle 保持默认即可。       |

### 转写示例

#### affine：是否进行仿射变换

```python
# 当 PyTorch 的 affine 为`False`，表示 weight 和 bias 不进行更新，torch 写法
torch.nn.modules.batchnorm._BatchNorm(num_features, affine=False)

# paddle 写法
paddle.nn.layer.norm._BatchNormBase(num_features, weight_attr=False, bias_attr=False)

# 当 PyTorch 的 affine 为`True`，torch 写法
torch.nn.modules.batchnorm._BatchNorm(num_features, affine=True)

# paddle 写法
paddle.nn.layer.norm._BatchNormBase(num_features)
```
