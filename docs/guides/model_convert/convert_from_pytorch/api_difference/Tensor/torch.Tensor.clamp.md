## [ 仅参数名不一致 ] torch.Tensor.clamp

### [torch.clamp](https://pytorch.org/docs/stable/generated/torch.clamp.html?highlight=clamp#torch.clamp)

```python
torch.clamp(input, min=None, max=None, *, out=None)
```

### [paddle.clip](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/clip_cn.html)

```python
paddle.clip(x, min=None, max=None, name=None)
```

两者功能一致，仅参数名不一致，具体如下：
### 参数映射
| PyTorch | PaddlePaddle | 备注                                               |
|---------|--------------| -------------------------------------------------- |
| input   | x            | 输入的 Tensor，仅参数名不一致。               |
| min     | min          | 裁剪的最小值，输入中小于该值的元素将由该元素代替。            |
| max     | max          | 裁剪的最大值，输入中大于该值的元素将由该元素代替。            |
