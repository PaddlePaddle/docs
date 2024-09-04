## [torch 参数更多]torch.Tensor.histogram

### [torch.Tensor.histogram](https://pytorch.org/docs/stable/generated/torch.Tensor.histogram.html#torch.Tensor.histogram)

```python
torch.Tensor.histogram(bins, *, range=None, weight=None, density=False)
```

### [paddle.Tensor.histogram](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#histogram-bins-100-min-0-max-0)

```python
paddle.Tensor.histogram(bins=100, min=0, max=0, weight=None, density=False)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                                                                                               |
| ------- | ------------ | -------------------------------------------------------------------------------------------------- |
| bins    | bins         | 直方图 bins(直条)的个数。                                                                          |
| range   | min, max     | PyTorch 为 bins 的范围，类型为 float，Paddle 为 range 的下边界，上边界，类型为 int，需要转写。 |
| weight  | weight       | 权重 Tensor，维度和 input 相同。    |
| density | density      | 表示直方图返回值是 count 还是归一化的频率，默认值 False 表示返回的是 count。  |

### 转写示例

#### range 参数：bins 的范围

```python
# PyTorch 写法:
x = torch.tensor([1., 2, 1])
hist, bin_edges = x.histogram(bins=5, range=(0., 3.))

# Paddle 写法:
x = paddle.to_tensor([1, 2, 1])
hist, bin_edges = x.histogram(bins=5, min=0, max=3), x.histogram_bin_edges(bins=5, min=0, max=3)
```
