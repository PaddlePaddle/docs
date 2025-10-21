## [ 仅 API 调用方式不一致 ]torch.median
### [torch.median](https://pytorch.org/docs/stable/generated/torch.median.html?highlight=median#torch.median)
```python
torch.median(input,
             dim=-1,
             keepdim=False,
             *,
             out=None)
```

### [paddle.median](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/median_cn.html#median)
```python
paddle.median(x, axis=None, keepdim=False, mode='avg', name=None)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 转写示例
#### out：指定输出
```python
# PyTorch 写法
torch.median([3, 5], out=y)

# Paddle 写法
paddle.assign(paddle.median([3, 5], mode='min'), y)
```
