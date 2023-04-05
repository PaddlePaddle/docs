## [ 参数完全一致 ]torch.Tensor.det

### [torch.Tensor.det](https://pytorch.org/docs/stable/generated/torch.Tensor.det.html?highlight=det#torch.Tensor.det)

```python
torch.Tensor.det()
```

### [paddle.linalg.det](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/linalg/det_cn.html#det)

```python
paddle.linalg.det(x)
```

两者功能参数完全一致，其中 torch 是类成员函数，而 paddle 是非类成员函数，因此输入参数 `x` 不进行对比，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                                                                                     |
| ------- | ------------ | ---------------------------------------------------------------------------------------- |
| -       | x            | 输入一个或批量矩阵。x 的形状应为 [*, M, M]，其中 * 为零或更大的批次维度，仅参数名不同。 |
