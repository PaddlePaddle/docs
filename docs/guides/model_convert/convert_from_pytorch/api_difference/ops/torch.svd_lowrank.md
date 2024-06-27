## [ 仅参数名不一致 ] torch.svd_lowrank

### [torch.svd_lowrank](https://pytorch.org/docs/stable/generated/torch.svd_lowrank.html?highlight=torch+svd_lowrank#torch.svd_lowrank)

```python
torch.svd_lowrank(A, q=6, niter=2, M=None)
```

### [paddle.linalg.svd_lowrank](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/linalg/svd_lowrank_cn.html)

```python
paddle.linalg.svd_lowrank(x, q=None, niter=2, M=None, name=None)
```

两者功能一致，仅参数名不一致，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| A          | x            | 表示输入 Tensor，仅参数名不一致。                           |
| q          | q            | 表示输入 Tensor 略高估计秩。                                |
| niter      | niter        | 表示子空间进行迭代的数量。                                  |
| M          | M            | 表示输入 Tensor 的平均值矩阵。                              |
