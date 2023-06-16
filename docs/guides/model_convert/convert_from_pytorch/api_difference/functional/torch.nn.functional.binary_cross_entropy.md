## [ 参数不一致 ] torch.nn.functional.binary_cross_entropy

### [torch.nn.functional.binary_cross_entropy](https://pytorch.org/docs/2.0/generated/torch.nn.functional.binary_cross_entropy.html?highlight=binary_cross_entropy#torch.nn.functional.binary_cross_entropy)

```python
torch.nn.functional.binary_cross_entropy(input, target, size_average=None, reduce=None, reduction='mean')
```

### [paddle.nn.functional.binary_cross_entropy](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/functional/binary_cross_entropy_cn.html#binary-cross-entropy)

```python
paddle.nn.functional.binary_cross_entropy(input, label, weight=None, reduction='mean', name=None)
```

两者功能一致，torch 参数多，具体如下：
### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input         | input        | 表示输入的 Tensor                                       |
| target        | label        | 标签                                                   |
| weight        | weight       | 指定每个 batch 的权重                                      |
| size_average  | -            | 已废弃，和 reduce 组合决定损失计算方式                      |
| reduce        | -            | 已废弃，和 size_average 组合决定损失计算方式                |
| reduction     | reduction    | 输出结果的计算方式                                       |


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
