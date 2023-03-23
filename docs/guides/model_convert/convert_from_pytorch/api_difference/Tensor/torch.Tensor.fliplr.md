## [ 仅 paddle 参数更多 ]torch.Tensor.fliplr

### [torch.Tensor.fliplr](https://pytorch.org/docs/stable/generated/torch.Tensor.fliplr.html?highlight=fliplr#torch.Tensor.fliplr)

```python
Tensor.fliplr()
```

### [paddle.Tensor.flip](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#flip-axis-name-none)

```python
Tensor.flip(axis, name=None)
```

两者功能一致，其中 Paddle 相比 Pytorch 支持更多其他参数，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                                                       |
| ------- | ------------ | ---------------------------------------------------------- |
| -       | axis         | 指定进行翻转的轴，Pytorch 无此参数，Paddle 中可以指定 `axis=1` 来对应 Pytorch。|

### 转写示例

```Python
x = np.random.random((3, 3)).astype("float32")
print(x)
# [[0.2585641  0.87252396 0.09721304]
#  [0.16210657 0.33419025 0.35641533]
#  [0.4821827  0.97285974 0.01860277]]

# torch 版本直接向上/下翻转张量
torch_x = torch.tensor(x)
print(torch_x.flipud())
# tensor([[0.0972, 0.8725, 0.2586],
#         [0.3564, 0.3342, 0.1621],
#         [0.0186, 0.9729, 0.4822]])
# Paddle 版本需要指定 axis=1
paddle_x = paddle.to_tensor(x)
print(paddle_x.flip(axis=1))
# Tensor(shape=[3, 3], dtype=float32, place=Place(cpu), stop_gradient=True,
#        [[0.09721304, 0.87252396, 0.25856411],
#         [0.35641533, 0.33419025, 0.16210657],
#         [0.01860277, 0.97285974, 0.48218271]])
```
