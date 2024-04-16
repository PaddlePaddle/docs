## [ 仅参数名不一致 ]torch.diagflat
### [torch.diagflat](https://pytorch.org/docs/stable/generated/torch.diagflat.html?highlight=diagflat#torch.diagflat)

```python
torch.diagflat(input,
               offset=0)
```

### [paddle.diagflat](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/diagflat_cn.html#diagflat)

```python
paddle.diagflat(x,
                offset=0,
                name=None)
```

两者功能一致且参数用法一致，仅参数名不一致，具体如下：
### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| <font color='red'> input </font> | <font color='red'> x </font> | 表示输入的 Tensor ，仅参数名不一致。  |
| offset | offset | 表示对角线偏移量。  |
