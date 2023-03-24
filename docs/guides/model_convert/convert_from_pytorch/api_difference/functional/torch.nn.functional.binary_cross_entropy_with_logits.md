## torch.nn.functional.binary_cross_entropy_with_logits

### [torch.nn.functional.binary_cross_entropy_with_logits](https://pytorch.org/docs/stable/generated/torch.nn.functional.binary_cross_entropy_with_logits.html?highlight=binary_cross_entropy_with_logits#torch.nn.functional.binary_cross_entropy_with_logits)

```python
torch.nn.functional.binary_cross_entropy_with_logits(input, target, weight=None, size_average=None, reduce=None, reduction='mean', pos_weight=None)
```

### [paddle.nn.functional.binary_cross_entropy_with_logits](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/functional/binary_cross_entropy_with_logits_cn.html)

```python
paddle.nn.functional.binary_cross_entropy_with_logits(logit, label, weight=None, reduction='mean', pos_weight=None, name=None)
```

两者功能一致，torch 参数多，具体如下：
### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input         | logit        | 表示输入的 Tensor                                       |
| target        | label        | 标签                                                   |
| weight        | weight       | 类别权重                                                |
| size_average  | -            | 已废弃，和 reduce 组合决定损失计算方式                      |
| reduce        | -            | 已废弃，和 size_average 组合决定损失计算方式                |
| reduction     | reduction    | 输出结果的计算方式                                       |
| pos_weight    | pos_weight   | 正类的权重                                              |

### 转写示例

```python
# Pytorch 的 size_average、reduce 参数转为 Paddle 的 reduction 参数
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
