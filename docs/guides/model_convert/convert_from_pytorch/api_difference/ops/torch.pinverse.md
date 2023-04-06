## [ 仅参数名不一致 ]torch.pinverse
### [torch.pinverse](https://pytorch.org/docs/stable/generated/torch.pinverse.html?highlight=pinverse#torch.pinverse)

```python
torch.pinverse(input,
               rcond=1e-15)
```

### [paddle.linalg.pinv](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/linalg/pinv_cn.html#pinv)

```python
paddle.linalg.pinv(x,
                   rcond=1e-15,
                   hermitian=False,
                   name=None)
```

其中 Paddle 相比 Pytorch 支持更多其他参数，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input         | x            | 输入的 Tensor，仅参数名不一致。                   |
| rcond         | rcond        | 奇异值（特征值）被截断的阈值，仅参数名不一致。        |
| -             | hermitian    | 是否为 hermitian 矩阵或者实对称矩阵，Pytorch 无，Paddle 保持默认即可。|
