## [ 参数不一致 ]torch.Tensor.view

### [torch.Tensor.view](https://pytorch.org/docs/stable/generated/torch.Tensor.view.html?highlight=view#torch.Tensor.view)

```python
torch.Tensor.view(*shape)
torch.Tensor.view(dtype)
```

### [paddle.Tensor.view](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#view-x-shape-or-dtype-name-none)

```python
paddle.Tensor.view(shape_or_dtype, name=None)
```

两者功能一致, 但 pytorch 的 `*shape` 和 paddle 的 `shape_or_dtype` 参数用法不一致，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                           |
| ------- | ------------ | ------------------------------ |
| *shape   | shape_or_dtype         | 指定的维度。 Pytorch 参数 shape 既可以是可变参数，也可以是 list/tuple/torch.Size/dtype 的形式， Paddle 参数 shape_or_dtype 为 list/tuple/dtype 的形式。需要进行转写。 |

### 转写示例

```python
# Pytorch 写法
x = torch.randn(4, 4)
x.view(2, 2, 4)

# Paddle 写法
x = paddle.randn(4, 4)
x.view([2, 2, 4])
```
