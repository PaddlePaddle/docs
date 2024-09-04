## [ 返回参数类型不一致]torch.histogram

### [torch.histogram](https://pytorch.org/docs/stable/generated/torch.histogram.html#torch.histogram)

```python
torch.histogram(input, bins, *, range=None, weight=None, density=False, out=None)
```

### [paddle.histogram](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/histogram_cn.html)

```python
paddle.histogram(input, bins=100, min=0, max=0, weight=None, density=False, name=None)
```

PyTorch 相比 Paddle 支持更多其他参数，返回参数 Tensor 数量不一致，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                                                                                               |
| ------- | ------------ | -------------------------------------------------------------------------------------------------- |
| input   | input        | 输入 Tensor。                                                                                      |
| bins    | bins         | 直方图 bins(直条)的个数。                                                                          |
| range   | min, max     | PyTorch 为 bins 的范围，类型为 float，Paddle 为 range 的下边界，上边界，类型为 int，需要转写。 |
| weight  | weight       | 权重 Tensor，维度和 input 相同。                                                              |
| density | density      | 表示直方图返回值是 count 还是归一化的频率，默认值 False 表示返回的是 count。                            |
| 返回值  | 返回值       | PyTorch 返回 hist 和 bin_edges，paddle.histogram 返回 hist，paddle.histogram_bin_edges 返回 bin_edges，需要转写。                                 |

### 转写示例

#### range 参数：bins 的范围；返回值

```python
# PyTorch 写法:
x = torch.tensor([1., 2, 1])
hist, bin_edges = torch.histogram(x, bins=5, range=(0., 3.))

# Paddle 写法:
x = paddle.to_tensor([1, 2, 1])
hist, bin_edges = paddle.histogram(x, bins=5, min=0, max=3), paddle.histogram_bin_edges(x, bins=5, min=0, max=3)
```
