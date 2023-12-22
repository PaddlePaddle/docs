## [ torch 参数更多 ]torch.nn.functional.fractional_max_pool2d

### [torch.nn.functional.fractional_max_pool2d](https://pytorch.org/docs/stable/generated/torch.nn.functional.fractional_max_pool2d.html#torch-nn-functional-fractional-max-pool2d)

```python
torch.nn.functional.fractional_max_pool2d(input, kernel_size, output_size=None, output_ratio=None, return_indices=False, _random_samples=None)
```

### [paddle.nn.functional.fractional_max_pool2d](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/functional/fractional_max_pool2d_cn.html)

```python
paddle.nn.functional.fractional_max_pool2d(x, output_size, random_u=None, return_mask=False, name=None)
```

PyTorch 参数更多，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input         | x            | 表示输入的 Tensor ，仅参数名不一致。                        |
| kernel_size   | -            | Paddle 内部推算核大小，不需要此参数。                       |
| output_ratio  | -            | Paddle 根据 output_size 推算输出比例，不需要此参数。        |
| return_indices | return_mask | 是否返回最大值索引，仅参数名不一致。                         |
| _random_samples | random_u   | 随机数，PyTorch 为随机数列表，Paddle 为单个随机数。功能一致。  |


### 转写示例

```python
# Pytorch 写法
torch.nn.functional.fractional_max_pool2d(input, 2, output_size=[3, 3], return_indices=True)

# Paddle 写法
paddle.nn.functional.fractional_max_pool2d(x, [3, 3], return_mask=True)
```
