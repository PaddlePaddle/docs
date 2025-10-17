## [ 仅参数名不一致 ]torch.kthvalue
### [torch.kthvalue](https://pytorch.org/docs/stable/generated/torch.kthvalue.html?highlight=kthvalue#torch.kthvalue)
```python
torch.kthvalue(input,
               k,
               dim=None,
               keepdim=False,
               *,
               out=None)
```

### [paddle.kthvalue](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/kthvalue_cn.html)
```python
paddle.kthvalue(x,
                k,
                axis=None,
                keepdim=False,
                name=None)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input         | x            | 表示输入的 Tensor ，仅参数名不一致。                   |
| k         | k           | 表示需要沿轴查找的第 k 小值。                   |
| dim         | axis            | 指定对输入 Tensor 进行运算的轴，仅参数名不一致。                   |
| keepdim         | keepdim            | 是否在输出 Tensor 中保留减小的维度。                   |
| out         | -            | 表示输出的 Tensor ， Paddle 无此参数，需要转写 。                   |
