## [ 参数完全一致 ] torch.Tensor.clip

### [torch.Tensor.clip](https://pytorch.org/docs/stable/generated/torch.Tensor.clip.html?highlight=clip#torch.Tensor.clip)

```python
torch.Tensor.clip(min=None, max=None)
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
