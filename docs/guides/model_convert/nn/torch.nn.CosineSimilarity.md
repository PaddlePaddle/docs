## torch.nn.CosineSimilarity
### [torch.nn.CosineSimilarity](https://pytorch.org/docs/stable/generated/torch.nn.CosineSimilarity.html?highlight=nn+cosinesimilarity#torch.nn.CosineSimilarity)

```python
torch.nn.CosineSimilarity(dim=1,
                            eps=1e-08)
```

### [paddle.nn.CosineSimilarity](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/CosineSimilarity_cn.html#cosinesimilarity)

```python
paddle.nn.CosineSimilarity(axis=1,
                            eps=1e-8)
```
### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| dim           | axis         | 指定计算的维度，会在该维度上计算余弦相似度，默认值为 1。        |
