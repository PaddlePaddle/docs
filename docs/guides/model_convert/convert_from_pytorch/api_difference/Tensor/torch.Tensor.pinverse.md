## [ 仅 Paddle 参数更多 ]torch.Tensor.pinverse
### [torch.Tensor.pinverse](https://pytorch.org/docs/stable/generated/torch.Tensor.pinverse.html#torch.Tensor.pinverse)

```python
torch.Tensor.pinverse()
```

### [paddle.linalg.pinv](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/linalg/pinv_cn.html#pinv)

```python
paddle.linalg.pinv(x,
                   rcond=1e-15,
                   hermitian=False,
                   name=None)
```

其中 Paddle 相比 Pytorch 支持更多参数，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| -         | rcond        | 奇异值（特征值）被截断的阈值，Pytorch 无此参数，Paddle 保持默认即可。        |
| -             | hermitian    | 是否为 hermitian 矩阵或者实对称矩阵，Pytorch 无此参数，Paddle 保持默认即可。|
