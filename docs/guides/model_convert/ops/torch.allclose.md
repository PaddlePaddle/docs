## torch.allclose
### [torch.allclose](https://pytorch.org/docs/stable/generated/torch.allclose.html?highlight=allclose#torch.allclose)

```python
torch.allclose(input,
                other,
                rtol=1e-05,
                atol=1e-08,
                equal_nan=False)
```

### [paddle.allclose](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/allclose_cn.html#allclose)

```python
paddle.allclose(x,
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
