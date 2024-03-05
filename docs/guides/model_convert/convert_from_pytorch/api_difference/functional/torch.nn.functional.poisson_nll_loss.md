## [ 参数不一致 ]torch.nn.functional.poisson_nll_loss

### [torch.nn.functional.poisson\_nll\_loss](https://pytorch.org/docs/stable/generated/torch.nn.functional.poisson_nll_loss.html)

```python
torch.nn.functional.poisson_nll_loss(input, target, log_input=True, full=False, size_average=None, eps=1e-08, reduce=None, reduction='mean')
```

### [paddle.nn.functional.poisson\_nll\_loss](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/functional/poisson_nll_loss_cn.html#poisson-nll-loss)

```python
paddle.nn.functional.poisson_nll_loss(input, label, log_input=True, full=False, epsilon=1e-8, reduction='mean', name=None)
```

其中 PyTorch 和 Paddle 功能一致，仅参数名不一致，具体如下：

### 参数映射

| PyTorch      | PaddlePaddle | 备注 |
| ------------ | ------------ | -- |
| input        | input        | 输入 Tensor。 |
| target       | label        | 标签 Tensor，仅参数名不一致。 |
| log_input    | log_input    | 输入是否为对数函数映射后结果。 |
| full         | full         | 是否在损失计算中包括 Stirling 近似项。 |
| size_average | -            | PyTorch 已弃用， Paddle 无此参数，需要转写。                  |
| eps          | epsilon      | 在 log_input 为 True 时使用的常数小量，仅参数名不一致。 |
| reduce       | -            | PyTorch 已弃用， Paddle 无此参数，需要转写。                  |
| reduction    | reduction    | 指定应用于输出结果的计算方式。 |

### 转写示例

#### size_average：做 reduce 的方式
```python
# PyTorch 的 size_average、reduce 参数转为 Paddle 的 reduction 参数
if size_average is None:
    size_average = True
if reduce is None:
    reduce = True

if size_average and reduce:
    reduction = 'mean'
elif reduce:
    reduction = 'sum'
else:
    reduction = 'none'
```
