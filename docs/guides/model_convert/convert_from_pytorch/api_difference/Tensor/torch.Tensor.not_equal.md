## [ 一致的参数 ] torch.Tensor.not_equal
### [torch.Tensor.not_equal](https://pytorch.org/docs/1.13/generated/torch.Tensor.not_equal.html)

```python
torch.Tensor.not_equal(other)
```

### [paddle.Tensor.not_equal](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/not_equal_cn.html)

```python
paddle.Tensor.not_equal(y)
```

两者功能一致，逐元素比较 Tensor 和 y 是否相等，相同位置的元素不相同则返回 True，否则返回 False。
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| other          | y         | 被比较的矩阵                                     |
