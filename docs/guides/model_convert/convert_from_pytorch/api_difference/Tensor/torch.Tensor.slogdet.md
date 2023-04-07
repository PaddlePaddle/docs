## [ 仅 paddle 参数更多 ]torch.Tensor.slogdet

### [torch.Tensor.slogdet](https://pytorch.org/docs/stable/generated/torch.Tensor.slogdet.html?highlight=torch+tensor+slogdet#torch.Tensor.slogdet)

```python
torch.Tensor.slogdet()
```

### [paddle.linalg.slogdet](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/linalg/slogdet_cn.html)

```python
paddle.linalg.slogdet(x)
```

两者功能一致，其中 Paddle 相比 Pytorch 支持更多其他参数，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                                                 |
|---------|--------------| ---------------------------------------------------- |
| -       | x            | 输入一个或批量矩阵。x 的形状应为 [*, M, M]，其中 * 为零或更大的批次维度，数据类型支持 float32、float64。|
