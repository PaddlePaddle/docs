## torch.sum
### [torch.sum](https://pytorch.org/docs/stable/generated/torch.sum.html?highlight=sum#torch.sum)

```python
torch.sum(input,
          dim=None,
          keepdim=False,
          dtype=None)
```

### [paddle.sum](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/sum_cn.html#sum)

```python
paddle.sum(x,
           axis=None,
           dtype=None,
           keepdim=False,
           name=None)
```

两者功能一致且参数用法一致，仅参数名不同，具体如下：
### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input         | x            | 输入的 Tensor。                   |
| dim           | axis         | 求和运算的维度。 |
