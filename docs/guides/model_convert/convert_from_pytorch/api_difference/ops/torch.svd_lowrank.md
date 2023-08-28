## [ torch 参数更多 ] torch.svd_lowrank

### [torch.svd_lowrank](https://pytorch.org/docs/stable/generated/torch.svd_lowrank.html?highlight=torch+svd_lowrank#torch.svd_lowrank)

```python
torch.svd_lowrank(A, q=6, niter=2, M=None)
```

### [paddle.linalg.svd](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/linalg/svd_cn.html#svd)

```python
paddle.linalg.svd(x, full_matrics=False, name=None)
```

Pytorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| A          | x            | 表示输入 Tensor，仅参数名不一致。                           |
| q          | -            | 表示输入 Tensor 略高估计秩。 Paddle 无此参数，暂无转写方式。   |
| niter          | -            | 表示子空间进行迭代的数量。Paddle 无此参数，暂无转写方式。    |
| M          | -            | 表示输入 Tensor 的平均 size。 Paddle 无此参数，暂无转写方式。   |
| -          | full_matrics            | 表示是否计算完整的 U 和 V 矩阵。 PyTorch 无此参数，Paddle 保持默认即可。       |
