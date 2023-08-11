## [ 仅参数名不一致 ]torch.bincount

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

两者功能一致且参数用法一致，仅参数名不一致，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| <font color='red'>input</font>| <font color='red'>x</font> | 表示输入的 Tensor ，仅参数名不一致。  |
| weights        | weights            | 表示输入 Tensor 中每个元素的权重。                   |
| minlength        | minlength            | 表示输出 Tensor 的最小长度。                   |
