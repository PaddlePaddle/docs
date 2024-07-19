## [ torch 参数更多 ]torch.nn.functional.fractional_max_pool2d

### [torch.nn.functional.fractional_max_pool2d](https://pytorch.org/docs/stable/generated/torch.nn.functional.fractional_max_pool2d.html#torch-nn-functional-fractional-max-pool2d)

```python
torch.nn.functional.fractional_max_pool2d(input, kernel_size, output_size=None, output_ratio=None, return_indices=False, _random_samples=None)
```

### [paddle.nn.functional.fractional_max_pool2d](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/functional/fractional_max_pool2d_cn.html)

```python
paddle.nn.functional.fractional_max_pool2d(x, output_size, kernel_size=None, random_u=None, return_mask=False, name=None)
```

PyTorch 参数更多，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input         | x            | 表示输入的 Tensor 。仅参数名不一致。                        |
| kernel_size   | kernel_size  | 表示核大小。参数完全一致。                                 |
| output_size   | output_size  | 表示目标输出尺寸，PyTorch 为可选参数，Paddle 为必选参数，Paddle 与 PyTorch 默认值不同。PyTorch 的 output_size 与 output_ratio 输入二选一，如不输入 output_size，则必须输入 output_ratio，此时需要转写。转写方式与下文 output_ratio 一致。 |
| output_ratio  | -            | 表示目标输出比例。Paddle 无此参数，需要转写。                |
| return_indices | return_mask | 表示是否返回最大值索引。仅参数名不一致。                      |
| _random_samples | random_u   | 表示随机数。PyTorch 以列表形式的 Tensor 方式传入，Paddle 以 float 的方式传入，如果 PyTorch 的多个随机数相同，需要转写，如果 PyTorch 的多个随机数不同，暂无转写方式。  |

### 转写示例

#### output_ratio:目标输出比例

```python
# 假设 intput 的 with=7, height=7，
# output_ratio = 0.75, 则目标 output 的 width = int(7*0.75) = 5, height = int(7*0.75) = 5
# Pytorch 写法
torch.nn.functional.fractional_max_pool2d(input, 2, output_ratio=[0.75, 0.75], return_indices=True)

# Paddle 写法
paddle.nn.functional.fractional_max_pool2d(x, output_size=[5, 5], kernel_size=2, return_mask=True)
```

#### _random_samples:随机数

```python
# Pytorch 写法
torch.nn.functional.fractional_max_pool2d(input, 2, output_size=[3, 3], return_indices=True, _random_samples=torch.tensor([[[0.3, 0.3]]]))

# Paddle 写法
paddle.nn.functional.fractional_max_pool2d(x, output_size=[3, 3], kernel_size=2, return_mask=True, random_u=0.3)
```
