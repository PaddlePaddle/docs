## [ 参数完全一致 ]torch.Tensor.logit_

### [torch.Tensor.logit_](https://pytorch.org/docs/stable/generated/torch.Tensor.logit_.html)

```python
torch.Tensor.logit_(input, eps=None, *, out=None)
```

### [paddle.Tensor.logit_](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/logit_cn.html)

```python
paddle.Tensor.logit_(x, eps=None, name=None)
```

两者功能一致，参数完全一致，具体如下：

### 参数映射

| PyTorch                             | PaddlePaddle | 备注                                                                    |
| ----------------------------------- | ------------ | ----------------------------------------------------------------------- |
| eps     | eps           | 将输入向量的范围控制在 [eps,1−eps]                        |
