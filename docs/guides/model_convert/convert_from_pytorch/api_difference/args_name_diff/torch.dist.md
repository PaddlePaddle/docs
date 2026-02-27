## [ 仅参数名不一致 ]torch.dist
### [torch.dist](https://docs.pytorch.org/docs/stable/generated/torch.dist.html#torch.dist)
```python
torch.dist(input,
           other,
           p=2)
```

### [paddle.dist](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/dist_cn.html#paddle.dist)
```python
paddle.dist(x,
            y,
            p=2)
```

两者功能一致且参数用法一致，仅参数名不一致，具体如下：
### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
|  input  |  x  | 表示输入的 Tensor ，仅参数名不一致。  |
|  other  |  y  | 表示输入的 Tensor ，仅参数名不一致。  |
| p | p | 表示需要计算的范数。 |
