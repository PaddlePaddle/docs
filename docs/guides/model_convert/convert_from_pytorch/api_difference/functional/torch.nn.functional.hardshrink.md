## [ 仅参数名不一致 ]torch.nn.functional.hardshrink

### [torch.nn.functional.hardshrink](https://pytorch.org/docs/stable/generated/torch.nn.functional.hardshrink.html?highlight=hardshrink#torch.nn.functional.hardshrink)

```python
torch.nn.functional.hardshrink(input,
                               lambd=0.5)
```

### [paddle.nn.functional.hardshrink](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/functional/hardshrink_cn.html)

```python
paddle.nn.functional.hardshrink(x,
                                threshold=0.5,
                                name=None)
```

两者功能一致，仅参数名不一致，具体如下：
### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input           | x           | 表示输入的 Tensor ，仅参数名不一致。               |
| lambd           | threshold           | 表示 hard_shrink 激活计算公式中的 threshold 值 ，仅参数名不一致。               |
