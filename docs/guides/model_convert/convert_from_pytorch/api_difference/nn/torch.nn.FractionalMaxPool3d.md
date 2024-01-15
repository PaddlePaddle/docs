## [ torch 参数更多 ]torch.nn.FractionalMaxPool3d

### [torch.nn.FractionalMaxPool3d](https://pytorch.org/docs/stable/generated/torch.nn.FractionalMaxPool3d.html#fractionalmaxpool3d)

```python
torch.nn.FractionalMaxPool3d(kernel_size, output_size=None, output_ratio=None, return_indices=False, _random_samples=None)
```

### [paddle.nn.FractionalMaxPool3D](https://www.paddlepaddle.org.cn/documentation/docs/en/develop/api/paddle/nn/FractionalMaxPool3D_cn.html)

```python
paddle.nn.FractionalMaxPool3D(output_size, kernel_size=None, random_u=None, return_mask=False, name=None)
```

PyTorch 参数更多，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| kernel_size   | kernel_size  | 表示核大小。与 PyTorch 默认值不同，（Paddle 可以不设置此参数）。 |
| output_size   | output_size  | 表示目标输出尺寸。参数完全一致。                                               |
| output_ratio  | -            | 表示目标输出比例。Paddle 无此参数，需要转写。                |
| return_indices | return_mask | 表示是否返回最大值索引。仅参数名不一致。                      |
| _random_samples | random_u   | 表示随机数。PyTorch 以列表形式的 Tensor 方式传入，Paddle 以 float 的方式传入，需要转写。  |

### 转写示例

#### output_ratio:目标输出比例

```python
# 假设 intput 的 depth=7, with=7, height=7，
# output_ratio = 0.75, 则目标 output 的 depth = int(7*0.75) = 5, width = int(7*0.75) = 5, height = int(7*0.75) = 5
# Pytorch 写法
torch.nn.FractionalMaxPool3d(2, output_ratio=[0.75, 0.75, 0.75], return_indices=True)

# Paddle 写法
paddle.nn.FractionalMaxPool2D(output_size=[5, 5, 5], kernel_size=2, return_mask=True)
```

#### _random_samples:随机数

```python
# Pytorch 写法
torch.nn.FractionalMaxPool3d(2, output_size=[3, 3, 3], return_indices=True, _random_samples=torch.tensor([[[0.3, 0.3, 0.3]]]))

# Paddle 写法
paddle.nn.FractionalMaxPool3D(output_size=[3, 3, 3], kernel_size=2, return_mask=True, random_u=0.3)
```
