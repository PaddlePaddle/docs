## [ 仅参数名不一致 ]torch.nn.functional.elu_

### [torch.nn.functional.elu_](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/functional/elu__cn.html)

```python
torch.nn.functional.elu_(input, alpha=1.0)
```

### [paddle.nn.functional.elu_](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/functional/relu__cn.html)

```python
paddle.nn.functional.elu_(x, alpha=1.0, name=None)
```

两者功能一致，仅参数名不一致，具体如下：

### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input           | x           | 表示输入的 Tensor ，仅参数名不一致。               |
| alpha           | alpha           | 表示 rule 激活公式中的超参数 。               |
