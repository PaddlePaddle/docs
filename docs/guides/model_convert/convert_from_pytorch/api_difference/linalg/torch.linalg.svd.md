## [ 仅参数默认值不一致 ] torch.linalg.svd

### [torch.linalg.svd](https://pytorch.org/docs/stable/generated/torch.linalg.svd.html?highlight=svd#torch.linalg.svd)

```python
torch.linalg.svd(A, full_matrices=True)
```

### [paddle.linalg.svd](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/linalg/svd_cn.html)

```python
paddle.linalg.svd(x, full_matrices=False, name=None)
```

两者功能一致且参数用法一致，仅参数默认值不一致，具体如下：

### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| A           | x           | 输入 Tensor，仅参数名不一致。               |
| full_matrices    | full_matrices  | 是否计算完整的 U 和 V 矩阵，Pytorch 为 True，Paddle 为 False，Paddle 需设置为与 Pytorch 一致。  |
