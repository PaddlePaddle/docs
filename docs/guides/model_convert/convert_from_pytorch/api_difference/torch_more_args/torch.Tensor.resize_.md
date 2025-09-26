## [ torch 参数更多 ]torch.Tensor.resize_
### [torch.Tensor.resize_](https://pytorch.org/docs/stable/generated/torch.Tensor.resize_.html)

```python
torch.Tensor.resize_(*sizes, memory_format=torch.contiguous_format)
```

### [paddle.Tensor.resize_](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#resize-shape-fill-zero-false-name-none)

```python
paddle.Tensor.resize_(shape, fill_zero=False, name=None)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| *sizes        | shape        | 设置的目标形状，torch 支持可变参数或 list/tuple，paddle 仅支持 list/tuple。对于可变参数的用法，需要转写。       |
| memory_format | -            | 设置后的内存形式，Paddle 无此参数，暂无转写方式。  |
| -             | fill_zero    | 当目标形状的元素个数大于 ``self`` 的元素个数时，是否用 0 填充新元素，若 False 则新元素的值不确定，PyTorch 无此参数，Paddle 保持默认即可。        |


### 转写示例

#### *sizes: 目标形状，可变参数用法
```python
# pytorch
x = torch.randn(2, 3)
x.resize_(4, 4)

# paddle
x = paddle.randn([2, 3])
x.resize_((4, 4))
```
