## [torch 参数更多]torch.nn.functional.hinge_embedding_loss

### [torch.nn.functional.hinge_embedding_loss](https://pytorch.org/docs/stable/generated/torch.nn.functional.hinge_embedding_loss.html#torch.nn.functional.hinge_embedding_loss)

```python
torch.nn.functional.hinge_embedding_loss(input, target, margin=1.0, size_average=None, reduce=None, reduction='mean')
```

### [paddle.nn.functional.hinge_embedding_loss](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/functional/hinge_embedding_loss_cn.html)

```python
paddle.nn.functional.hinge_embedding_loss(input, label, margin=1.0, reduction='mean', name=None)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch      | PaddlePaddle | 备注                                                                                       |
| ------------ | ------------ | ------------------------------------------------------------------------------------------ |
| input        | input        | 输入的 Tensor。                                                                            |
| target       | label        | 标签，仅参数名不一致。                                                                     |
| margin       | margin       | 当 label 为 -1 时，该值决定了小于 margin 的 input 才需要纳入 hinge embedding loss 的计算。 |
| size_average | -            | 已废弃，和 reduce 组合决定损失计算方式。                                                   |
| reduce       | -            | 已废弃，和 size_average 组合决定损失计算方式。                                             |
| reduction    | reduction    | 指定应用于输出结果的计算方式。                                                             |

### 转写示例

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
