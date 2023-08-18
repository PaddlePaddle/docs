## [ 参数完全一致 ] torch.Tensor.clamp

### [torch.Tensor.clamp](https://pytorch.org/docs/stable/generated/torch.Tensor.clamp.html?highlight=clamp#torch.Tensor.clamp)

```python
torch.Tensor.clamp(min=None, max=None)
```

### [paddle.Tensor.clip](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#clip-min-none-max-none-name-none)

```python
paddle.Tensor.clip(min=None, max=None, name=None)
```

两者功能一致，参数完全一致，具体如下：
### 参数映射
| PyTorch | PaddlePaddle | 备注                                               |
|---------|--------------| -------------------------------------------------- |
| min     | min          | 裁剪的最小值，输入中小于该值的元素将由该元素代替。            |
| max     | max          | 裁剪的最大值，输入中大于该值的元素将由该元素代替。            |
