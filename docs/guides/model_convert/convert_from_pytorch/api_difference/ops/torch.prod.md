## torch.prod
### [torch.prod](https://pytorch.org/docs/stable/generated/torch.prod.html?highlight=prod#torch.prod)


```python
torch.prod(input,
           dim=None,
           keepdim=False,
           dtype=None)
```

### [paddle.prod](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/prod_cn.html#prod)

```python
paddle.prod(x,
            axis=None,
            keepdim=False,
            dtype=None,
            name=None)
```

两者功能一致且参数用法一致，仅参数名不同，具体如下：
### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input         | x            | 输入的 Tensor。                   |
| dim           | axis         | 求乘积运算的维度。                  |
