## [ 仅 Paddle 参数更多 ] torch.Tensor.clamp_

### [torch.clamp_](https://pytorch.org/docs/stable/generated/torch.Tensor.clamp_.html?highlight=clamp_#torch.Tensor.clamp_)

```python
Tensor.clamp_(min=None, max=None)
```

### [paddle.clip_](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/clip__cn.html)

```python
paddle.clip_(x, min=None, max=None, name=None)
```

两者功能一致，仅 Paddle 参数更多，具体如下：
### 参数映射
| PyTorch | PaddlePaddle | 备注                        |
|---------|--------------|---------------------------|
|         | x            | 输入的 Tensor，PyTorch 无此参数， Paddle 保持默认即可。    |
| min     | min          | 裁剪的最小值，输入中小于该值的元素将由该元素代替。 |
| max     | max          | 裁剪的最大值，输入中大于该值的元素将由该元素代替。 |
