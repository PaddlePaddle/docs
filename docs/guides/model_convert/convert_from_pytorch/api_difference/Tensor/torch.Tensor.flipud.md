## [ 仅 paddle 参数更多 ]torch.Tensor.flipud

### [torch.Tensor.flipud](https://pytorch.org/docs/stable/generated/torch.Tensor.flipud.html?highlight=flipud#torch.Tensor.flipud)

```python
torch.Tensor.flipud()
```

### [paddle.Tensor.flip](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/Tensor_cn.html#flip-axis-name-none)

```python
paddle.Tensor.flip(axis, name=None)
```

两者功能一致，其中 Paddle 相比 Pytorch 支持更多其他参数，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                                                       |
| ------- | ------------ | ---------------------------------------------------------- |
| -       | axis         | 指定进行翻转的轴，Pytorch 无此参数，Paddle 中可以指定 `axis=0` 来对应 Pytorch。|

### 转写示例

```Python
# torch 版本直接向上/下翻转张量
torch_x = torch.randn(3, 4)
torch_x.flipud()

# paddle 版本直接向上/下翻转张量
paddle_x = paddle.randn([3, 4])
paddle_x.flip(axis=0)
```
