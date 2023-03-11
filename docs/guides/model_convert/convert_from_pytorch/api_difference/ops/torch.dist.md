## [ 仅参数名不一致 ]torch.dist
### [torch.dist](https://pytorch.org/docs/stable/generated/torch.dist.html?highlight=dist#torch.dist)

```python
torch.dist(input,
           other,
           p=2)
```

### [paddle.dist](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/dist_cn.html#dist)

```python
paddle.dist(x,
            y,
            p=2)
```

两者功能一致且参数用法一致，仅参数名不同，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| <font color='red'> input </font> | <font color='red'> x </font> | 表示输入的 Tensor ，仅参数名不一致。  |
| <font color='red'> other </font> | <font color='red'> y </font> | 表示输入的 Tensor ，仅参数名不一致。  |
| p | p | 表示需要计算的范数 |
