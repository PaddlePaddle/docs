## torch.bincount
### [torch.bincount](https://pytorch.org/docs/stable/generated/torch.bincount.html?highlight=bincount#torch.bincount)

```python
torch.bincount(input,
               weights=None,
               minlength=0)
```

### [paddle.bincount](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/bincount_cn.html#bincount)

```python
paddle.bincount(x,
                weights=None,
                minlength=0,
                name=None)
```

两者功能一致且参数用法一致，仅参数名不同，具体如下：
### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input        | x            | 输入的 Tensor。                   |
