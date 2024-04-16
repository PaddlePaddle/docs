## [ 仅 paddle 参数更多 ]torch.Tensor.fliplr

### [torch.Tensor.fliplr](https://pytorch.org/docs/stable/generated/torch.Tensor.fliplr.html?highlight=fliplr#torch.Tensor.fliplr)

```python
torch.Tensor.fliplr()
```

### [paddle.Tensor.flip](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#flip-axis-name-none)

```python
paddle.Tensor.flip(axis, name=None)
```

两者功能一致，其中 Paddle 相比 PyTorch 支持更多其他参数，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                                                       |
| ------- | ------------ | ---------------------------------------------------------- |
| -       | axis         | 指定进行翻转的轴，PyTorch 无此参数，Paddle 中可以指定 `axis=1` 来对应 PyTorch。|

### 转写示例

```Python
# torch 版本直接向左/右翻转张量
torch_x = torch.randn(3, 4)
torch_x.fliplr()

# paddle 版本直接向左/右翻转张量
paddle_x = paddle.randn([3, 4])
paddle_x.flip(axis=1)
```
