## [ 仅 paddle 参数更多 ]torch.Tensor.det

### [torch.Tensor.det](https://pytorch.org/docs/stable/generated/torch.Tensor.det.html?highlight=det#torch.Tensor.det)

```python
Tensor.det()
```

### [paddle.linalg.det](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/linalg/det_cn.html#det)

```python
paddle.linalg.det(x)
```

两者功能一致，其中 Paddle 相比 Pytorch 支持更多其他参数，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                                                                                     |
| ------- | ------------ | ---------------------------------------------------------------------------------------- |
| -       | x            | 输入一个或批量矩阵。x 的形状应为 [*, M, M]，其中 \* 为零或更大的批次维度，仅参数名不同。 |
