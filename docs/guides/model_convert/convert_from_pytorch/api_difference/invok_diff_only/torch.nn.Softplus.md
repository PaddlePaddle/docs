## [ 参数完全一致 ] torch.nn.Softplus

### [torch.nn.Softplus](https://pytorch.org/docs/stable/generated/torch.nn.Softplus.html)

```python
torch.nn.Softplus(beta=1, threshold=20)
```

### [paddle.nn.Softplus](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/Softplus_cn.html)

```python
paddle.nn.Softplus(beta=1, threshold=20, name=None)
```

两者功能一致，参数完全一致，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| beta          | beta         | Softplus 公式的 beta 值。                              |
| threshold     | threshold    | 恢复线性公式的阈值。                                    |
