## [ 仅 API 调用方式不一致 ]torch.utils.data.WeightedRandomSampler
### [torch.utils.data.WeightedRandomSampler](https://pytorch.org/docs/stable/data.html#torch.utils.data.WeightedRandomSampler)
```python
torch.utils.data.WeightedRandomSampler(weights,
                       num_samples,
                       replacement=True,
                       generator=None)
```

### [paddle.io.WeightedRandomSampler](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/io/WeightedRandomSampler_cn.html#paddle.io.WeightedRandomSampler)
```python
paddle.io.WeightedRandomSampler(weights,
                num_samples,
                replacement=True)
```

两者功能一致，但调用方式不一致，具体如下：

### 转写示例

```python
# PyTorch 写法
sampler = torch.utils.data.WeightedRandomSampler([0.5, 0.5], 20, True)

# Paddle 写法
sampler = paddle.io.WeightedRandomSampler([0.5, 0.5], 20, True)
```
