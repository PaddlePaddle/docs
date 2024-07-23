## [仅参数名不一致]torch.pca_lowrank

### [torch.pca_lowrank](https://pytorch.org/docs/stable/generated/torch.pca_lowrank.html#torch.pca_lowrank)

```python
torch.pca_lowrank(A, q=None, center=True, niter=2)
```

### [paddle.linalg.pca_lowrank](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/linalg/pca_lowrank_cn.html)

```python
paddle.linalg.pca_lowrank(x, q=None, center=True, niter=2, name=None)
```

两者功能一致，仅参数名不一致，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                                                                          |
| ------- | ------------ | ----------------------------------------------------------------------------- |
| A       | x            | 输入的需要进行线性主成分分析的一个或一批方阵，仅参数名不一致。 |
| q       | q            | 对输入 Tensor 的秩稍微高估的预估值。                                          |
| center  | center       | 是否对输入矩阵进行中心化操作。                                                |
| niter   | niter        | 表示子空间进行迭代的数量。                                                    |
