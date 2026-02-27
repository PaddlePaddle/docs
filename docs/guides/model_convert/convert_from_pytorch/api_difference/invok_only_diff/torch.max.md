## [ 仅 API 调用方式不一致 ]torch.max

### [torch.max](https://docs.pytorch.org/docs/stable/generated/torch.max.html#torch.max)

```python
torch.max(*args, **kwargs)
```

### [paddle.compat.max](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/compat/max_cn.html#id3)

```python
paddle.compat.max(*args, **kwargs)
```

两者功能和用法一致，但 API 路径不一致，只需修改 torch 前缀为 paddle.compat，具体如下：

### 转写示例

```python
# PyTorch 写法
result = torch.max(x)

# Paddle 写法
result = paddle.compat.max(x)

```
