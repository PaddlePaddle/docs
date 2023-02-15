## torch.isclose
### [torch.isclose](https://pytorch.org/docs/stable/generated/torch.isclose.html?highlight=isclose#torch.isclose)

```python
torch.isclose(input,
                other,
                rtol=1e-05,
                atol=1e-08,
                equal_nan=False)
```

### [paddle.isclose](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/isclose_cn.html#isclose)

```python
paddle.isclose(x,
                y,
                rtol=1e-05,
                atol=1e-08,
                equal_nan=False,
                name=None)
```
### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input        | x            | 输入的 Tensor。                   |
| other        | y            | 输入的 Tensor。                   |
