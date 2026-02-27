## [ 仅 API 调用方式不一致 ]torch.min

### [torch.min](https://docs.pytorch.org/docs/stable/generated/torch.min.html#torch.min)

```python
torch.min(*args, **kwargs)
```

### [paddle.compat.min](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/compat/min_cn.html#id3)

```python
paddle.compat.min(*args, **kwargs)
```

两者功能和用法一致，但 API 路径不一致，只需修改 torch 前缀为 paddle.compat，具体如下：

### 转写示例

```python
# PyTorch 写法
result = torch.min(x)

# Paddle 写法
result = paddle.compat.min(x)

```
