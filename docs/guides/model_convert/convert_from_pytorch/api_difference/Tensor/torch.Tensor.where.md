## [torch 参数更多]torch.Tensor.where

### [torch.Tensor.where](https://pytorch.org/docs/1.13/generated/torch.Tensor.where.html#torch.Tensor.where)

```python
torch.Tensor.where(condition, y)
```

### [paddle.Tensor.where](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#where-y-name-none)

```python
paddle.Tensor.where(y, name=None)
```

两者功能一致，torch 参数更多，具体如下：
### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| condition        | -            | 当 condition 为 true 时选择 Tensor 中元素，反之选择 y 中元素。                                     |
| y| y        | 在 pyTorch 中，当 condition 为 false 时，选择 y 中元素。而 paddle 是直接将 Tensor 作为 condition，如果 Tensor 元素小于 0，则选择 y 中的元素。       |


### 转写示例

```python
# torch 写法
import torch

a = torch.tensor([0, 1, 2])
b = torch.tensor([2, 3, 0])
c = a.where(a>0, b)
print(c)

# paddle 写法
import paddle
a = paddle.to_tensor([0, 1, 2])
b = paddle.to_tensor([2, 3, 0])
c = a.where(b)
print(c)
```
