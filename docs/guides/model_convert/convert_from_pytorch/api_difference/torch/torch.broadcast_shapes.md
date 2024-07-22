## [ 输入参数用法不一致 ]torch.broadcast_shapes

### [torch.broadcast_shapes](https://pytorch.org/docs/stable/generated/torch.broadcast_shapes.html#torch.broadcast_shapes)

```python
torch.broadcast_shapes(*shapes)
```

### [paddle.broadcast_shape](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/broadcast_shape_cn.html)

```python
paddle.broadcast_shape(x_shape, y_shape)
```

其中功能一致, 参数用法不一致，具体如下：

### 参数映射

| PyTorch  | PaddlePaddle     | 备注                                                                                                      |
| -------- | ---------------- | --------------------------------------------------------------------------------------------------------- |
| *shapes | x_shape, y_shape | 输入 Tensor 的 shape，输入参数数量不一致，PyTorch 可以输入多项，Paddle 输入为两项，需要多次调用进行转写。 |

### 转写示例

#### shapes 参数：输入 Tensor 的 shape

```python
# PyTorch 写法:
torch.broadcast_shapes(x1, x2, x3)

# Paddle 写法:
paddle.broadcast_shape(paddle.broadcast_shape(x1, x2), x3)
```
