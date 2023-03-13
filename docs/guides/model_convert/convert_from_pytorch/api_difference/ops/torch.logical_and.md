## torch.logical_and
### [torch.logical_and](https://pytorch.org/docs/1.13/generated/torch.logical_and.html?highlight=logical_and#torch.logical_and)

```python
torch.logical_and(input,
                  other,
                  *,
                  out=None)
```

### [paddle.logical_and](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/logical_and_cn.html#logical-and)

```python
paddle.logical_and(x,
                   y,
                   out=None,
                   name=None)
```

两者功能一致且参数用法一致，仅参数名不同，具体如下：
### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input         | x            | 输入的 Tensor。                                      |
| other         | y            | 输入的 Tensor。                                      |
