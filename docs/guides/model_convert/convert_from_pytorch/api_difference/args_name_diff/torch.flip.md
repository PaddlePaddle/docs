## [ 仅参数名不一致 ]torch.flip
### [torch.flip](https://pytorch.org/docs/stable/generated/torch.flip.html?highlight=flip#torch.flip)

```python
torch.flip(input,
           dims)
```

### [paddle.flip](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/flip_cn.html#flip)

```python
paddle.flip(x,
            axis,
            name=None)
```

两者功能一致且参数用法一致，仅参数名不一致，具体如下：
### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| <font color='red'> input </font> | <font color='red'> x </font> | 表示输入的 Tensor ，仅参数名不一致。  |
| <font color='red'> dims </font> | <font color='red'> axis </font> | 表示进行运算的轴，仅参数名不一致。  |
