## [ 组合替代实现 ]torch.inference_mode

### [torch.inference_mode](https://pytorch.org/docs/stable/generated/torch.inference_mode.html#torch.inference_mode)

```python
torch.inference_mode(mode=True)
```

Paddle 无此 API，需要组合实现。

### 转写示例

```python
# PyTorch 写法
@torch.inference_mode(False)

# Paddle 写法
@empty_decorator()


# PyTorch 写法
@torch.inference_mode(True)

# Paddle 写法
@paddle.no_grad()
```
