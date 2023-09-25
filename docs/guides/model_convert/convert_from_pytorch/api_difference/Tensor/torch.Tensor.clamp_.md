## [ 参数完全一致 ] torch.Tensor.clamp_

### [torch.Tensor.clamp_](https://pytorch.org/docs/stable/generated/torch.Tensor.clamp_.html?highlight=clamp_#torch.Tensor.clamp_)

```python
torch.Tensor.clamp_(min=None, max=None)
```

### [paddle.Tensor.clip_](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#id6)

```python
paddle.Tensor.clip_(min=None, max=None, name=None)
```

两者功能一致，参数完全一致，具体如下：
### 参数映射
| PyTorch | PaddlePaddle | 备注                        |
|---------|--------------|---------------------------|
| min     | min          | 裁剪的最小值，输入中小于该值的元素将由该元素代替。 |
| max     | max          | 裁剪的最大值，输入中大于该值的元素将由该元素代替。 |
