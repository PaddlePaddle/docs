## [ 仅 API 调用方式不一致 ]torch.nanmedian
### [torch.nanmedian](https://pytorch.org/docs/stable/generated/torch.nanmedian.html?highlight=nanmedian#torch.nanmedian)
```python
torch.nanmedian(input,
                dim=-1,
                keepdim=False,
                *,
                out=None)
```

### [paddle.nanmedian](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nanmedian_cn.html#nanmedian)
```python
paddle.nanmedian(x, axis=None, keepdim=False, mode='avg', name=None)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 转写示例
#### out：指定输出
```python
# PyTorch 写法
torch.nanmedian(a, -1, out=y)

# Paddle 写法
paddle.assign(paddle.nanmedian(a, -1, mode='min'), y)
```
