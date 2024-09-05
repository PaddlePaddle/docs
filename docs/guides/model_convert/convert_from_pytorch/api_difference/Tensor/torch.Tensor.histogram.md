## [ 组合替代实现 ]torch.Tensor.histogram

### [torch.Tensor.histogram](https://pytorch.org/docs/stable/generated/torch.Tensor.histogram.html#torch.Tensor.histogram)

```python
torch.Tensor.histogram(bins, *, range=None, weight=None, density=False)
```

### [paddle.Tensor.histogram](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#histogram-bins-100-min-0-max-0)

```python
paddle.Tensor.histogram(bins=100, min=0, max=0, weight=None, density=False)
```

其中 PyTorch 的 `range` 与 Paddle 用法不一致，需要转写；且返回参数 Tensor 数量不一致，需要通过 paddle.Tensor.histogram 和 paddle.Tensor.histogram_bin_edges 组合实现。具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                                                                                               |
| ------- | ------------ | -------------------------------------------------------------------------------------------------- |
| bins    | bins         | 直方图 bins(直条)的个数。                                                                          |
| range   | min, max     | PyTorch 为 bins 的范围，类型为 float，Paddle 为 range 的下边界，上边界，类型为 int，需要转写。 |
| weight  | weight       | 权重 Tensor，维度和 input 相同。    |
| density | density      | 表示直方图返回值是 count 还是归一化的频率，默认值 False 表示返回的是 count。  |
| 返回值  | 返回值       | PyTorch 返回 hist 和 bin_edges，paddle.Tensor.histogram 返回 hist，paddle.Tensor.histogram_bin_edges 返回 bin_edges，需要转写。                                 |

### 转写示例

#### range 参数：bins 的范围

```python
# PyTorch 写法:
x = torch.tensor([1., 2, 1])
hist, _ = x.histogram(bins=5, range=(0., 3.))

# Paddle 写法:
x = paddle.to_tensor([1, 2, 1])
hist = x.histogram(bins=5, min=0, max=3)
```

#### 返回值

```python
# PyTorch 写法:
x = torch.tensor([1., 2, 1])
hist, bin_edges = x.histogram(x, bins=5)

# Paddle 写法:
x = paddle.to_tensor([1, 2, 1])
hist, bin_edges = x.histogram(x, bins=5), x.histogram_bin_edges(x, bins=5)
```
