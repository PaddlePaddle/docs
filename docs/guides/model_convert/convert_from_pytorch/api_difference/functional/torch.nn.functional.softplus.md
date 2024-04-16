## [ 仅参数名不一致 ]torch.nn.functional.softplus

### [torch.nn.functional.softplus](https://pytorch.org/docs/stable/generated/torch.nn.functional.softplus.html?highlight=softplus#torch.nn.functional.softplus)

```python
torch.nn.functional.softplus(input,
                             beta=1,
                             threshold=20)
```

### [paddle.nn.functional.softplus](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/functional/softplus_cn.html)

```python
paddle.nn.functional.softplus(x,
                              beta=1,
                              threshold=20,
                              name=None)
```

两者功能一致，仅参数名不一致，具体如下：
### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input           | x           | 表示 Softplus 激活计算公式中的 beta 值 ，仅参数名不一致。               |
| beta           | beta           | 表示 Softplus 激活计算公式中的 threshold 值 。               |
| threshold           | threshold           | 表示输入的 Tensor 。               |
