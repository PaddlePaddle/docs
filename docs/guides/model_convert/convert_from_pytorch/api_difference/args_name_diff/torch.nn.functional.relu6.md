## [ 仅参数名不一致 ]torch.nn.functional.relu6

### [torch.nn.functional.relu6](https://pytorch.org/docs/stable/generated/torch.nn.functional.relu6.html?highlight=relu6#torch.nn.functional.relu6)

```python
torch.nn.functional.relu6(input, inplace=False)
```

### [paddle.nn.functional.relu6](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/functional/relu6_cn.html)

```python
paddle.nn.functional.relu6(x, name=None)
```

两者功能一致，仅参数名不一致，具体如下：
### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input         | x           | 表示输入的 Tensor ，仅参数名不一致。               |
| inplace       | -            | 表示在不更改变量的内存地址的情况下，直接修改变量的值，Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。 |
