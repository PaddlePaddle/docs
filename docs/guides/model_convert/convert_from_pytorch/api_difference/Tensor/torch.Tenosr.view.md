## [ 无参数 ] torch.Tensor.view

### [torch.Tensor.view](https://pytorch.org/docs/stable/generated/torch.Tensor.view.html)

```python
torch.Tensor.view(*shape) 
```

### [paddle.view](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/view_cn.html)

```python
paddle.view(x, shape_or_dtype, name=None)
```

两者功能一致，但 pytorch 的 `shape` 和 paddle 的 `shape_or_dtype` 参数用法不一致，具体如下：

### 参数映射

| Pytorch | PaddlePaddle | 备注                                                         |
| ------- | ------------ | :----------------------------------------------------------- |
| *shape   | shape_or_dtype | 目标尺寸， Pytorch 参数 shape 既可以是可变参数，也可以是 list/tuple/tensor 的形式， Paddle 参数 shape 为 list/tuple/tensor 的形式，同时也可以是类型参数，PyTorch是Paddle的子集。 |

转写示例

#### ***reps: 维度复制次数**

```
# Pytorch 写法
x = torch.tensor([1, 2, 3])
x.shape(2,3)
# Paddle 写法
y= paddle.to_tensor([1, 2, 3])
paddle.view(y, [2,3])
```
