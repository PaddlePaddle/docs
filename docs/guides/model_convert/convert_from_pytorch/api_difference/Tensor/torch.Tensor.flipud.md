## [ 仅 paddle 参数更多 ]torch.Tensor.flipud

### [torch.Tensor.flipud](https://pytorch.org/docs/stable/generated/torch.Tensor.flipud.html?highlight=flipud#torch.Tensor.flipud)

```python
Tensor.flipud()
```

### [paddle.Tensor.flip](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/Tensor_cn.html#flip-axis-name-none)

```python
Tensor.flip(axis, name=None)
```

两者功能一致，其中 Paddle 相比 Pytorch 支持更多其他参数，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                                                       |
| ------- | ------------ | ---------------------------------------------------------- |
| -       | axis         | 指定进行翻转的轴，Pytorch 无此参数，Paddle 中可以指定 `axis=0` 来对应 Pytorch。|

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
# tensor([[0.4822, 0.9729, 0.0186],
#         [0.1621, 0.3342, 0.3564],
#         [0.2586, 0.8725, 0.0972]])
# Paddle 版本需要指定 axis=0
paddle_x = paddle.to_tensor(x)
print(paddle_x.flip(axis=0))
# Tensor(shape=[3, 3], dtype=float32, place=Place(cpu), stop_gradient=True,
#        [[0.48218271, 0.97285974, 0.01860277],
#         [0.16210657, 0.33419025, 0.35641533],
#         [0.25856411, 0.87252396, 0.09721304]])
```
