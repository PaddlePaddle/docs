## [ torch 参数更多 ]torch.nn.FractionalMaxPool3d

### [torch.nn.FractionalMaxPool3d](https://pytorch.org/docs/stable/generated/torch.nn.FractionalMaxPool3d.html#fractionalmaxpool3d)

```python
torch.nn.FractionalMaxPool3d(kernel_size, output_size=None, output_ratio=None, return_indices=False, _random_samples=None)
```

### [paddle.nn.FractionalMaxPool3D](https://www.paddlepaddle.org.cn/documentation/docs/en/develop/api/paddle/nn/FractionalMaxPool3D_cn.html)

```python
paddle.nn.FractionalMaxPool3D(output_size, random_u=None, return_mask=False, name=None)
```

PyTorch 参数更多，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| kernel_size   | -            | Paddle 内部推算核大小，不需要此参数。                       |
| output_ratio  | -            | Paddle 根据 output_size 推算输出比例，不需要此参数。        |
| return_indices | return_mask | 是否返回最大值索引，仅参数名不一致。                         |
| _random_samples | random_u   | 随机数，PyTorch 为随机数列表，Paddle 为单个随机数。功能一致。  |


### 转写示例

```python
# Pytorch 写法
torch.nn.FractionalMaxPool3d(2, output_size=[3, 3, 3], return_indices=True)

# Paddle 写法
paddle.nn.FractionalMaxPool3D([3, 3, 3], return_mask=True)
```
