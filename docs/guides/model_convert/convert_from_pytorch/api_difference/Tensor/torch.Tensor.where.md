## [参数用法不⼀致]torch.Tensor.where

### [torch.Tensor.where](https://pytorch.org/docs/1.13/generated/torch.Tensor.where.html#torch.Tensor.where)

```python
torch.Tensor.where(condition, y)
```

### [paddle.Tensor.where](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#where-y-name-none)

```python
paddle.Tensor.where(x, y, name=None)
```

两者功能一致，仅参数用法不一致，具体如下：
### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| condition        | x            | 在 pytorch 中 condition 为判断条件，而在 paddle 中 condition 需要转写， 当 condition 为 true 时，选择 x 中元素。                                     |
| y| y        | 当 condition 为 false 时，选择 y 中元素，在 paddle 中 condition 需要转写。       |


### 转写示例

```python
# torch 写法
a = torch.tensor([0, 1, 2])
b = torch.tensor([2, 3, 0])
c = a.where(a > 0, b)

# paddle 写法
a = paddle.to_tensor([0, 1, 2])
b = paddle.to_tensor([2, 3, 0])
c = (a > 0).where(a, b)
```
