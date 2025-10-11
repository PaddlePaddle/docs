## [ paddle 参数更多 ]torch.Tensor.pinverse
### [torch.Tensor.pinverse](https://pytorch.org/docs/stable/generated/torch.Tensor.pinverse.html#torch.Tensor.pinverse)

```python
torch.Tensor.pinverse()
```

### [paddle.Tensor.pinv]()

```python
paddle.Tensor.pinv(rcond=1e-15,
                   hermitian=False,
                   name=None)
```

其中 Paddle 相比 PyTorch 支持更多参数，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| -         | rcond        | 奇异值（特征值）被截断的阈值，PyTorch 无此参数，Paddle 保持默认即可。        |
| -             | hermitian    | 是否为 hermitian 矩阵或者实对称矩阵，PyTorch 无此参数，Paddle 保持默认即可。|
