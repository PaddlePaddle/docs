## [ 仅参数名不一致 ]torch.cosine_similarity
### [torch.cosine_similarity](https://pytorch.org/docs/stable/generated/torch.nn.functional.cosine_similarity.html?highlight=cosine_similarity#torch.nn.functional.cosine_similarity)

```python
torch.cosine_similarity(x1, x2, dim=1, eps=1e-8)
```

### [paddle.nn.functional.cosine_similarity](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/functional/cosine_similarity_cn.html#cosine-similarity)

```python
paddle.nn.functional.cosine_similarity(x1, x2, axis=1, eps=1e-8)
```

两者功能一致，仅参数名不一致，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| x1          | x1         | 表示第一个输入的 Tensor 。                                     |
| x2          | x2         | 表示第二个输入的 Tensor 。                                     |
| dim          | axis         | 表示计算的维度，仅参数名不一致。                                     |
| eps          | eps         | 表示加到分母上的超参数 。                                     |
