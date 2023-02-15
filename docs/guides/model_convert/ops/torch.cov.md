## torch.cov
### [torch.cov](https://pytorch.org/docs/stable/generated/torch.cov.html?highlight=cov#torch.cov)

```python
torch.cov(input,
            correction=1,
            fweights=None,
            aweights=None)
```

### [paddle.linalg.cov](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/linalg/cov_cn.html#cov)

```python
paddle.linalg.cov(x,
                    rowvar=True,
                    ddof=True,
                    fweights=None,
                    aweights=None,
                    name=None)
```
### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input        | x            | 输入的 Tensor。                   |
| correction   | ddof         | 若为 True，返回无偏估计结果；若为 False，返回普通平均值计算结果。 |
