## [ 仅 API 调用方式不一致 ]torch.autograd.graph.saved_tensors_hooks

### [torch.autograd.graph.saved_tensors_hooks](https://pytorch.org/docs/stable/generated/torch.autograd.graph.html#torch.autograd.graph.saved_tensors_hooks)

```python
torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook)
```

### [paddle.autograd.saved_tensors_hooks](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/autograd/saved_tensors_hooks_cn.html#paddle/autograd/saved_tensors_hooks_cn#cn-api-paddle-autograd-saved_tensors_hooks)

```python
paddle.autograd.saved_tensors_hooks(pack_hook, unpack_hook)
```

两者功能一致，但调用方式不一致，具体如下：

### 转写示例

```python
# PyTorch 写法
with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
    y = torch.mul(a, b)

# Paddle 写法
with paddle.autograd.saved_tensors_hooks(pack_hook, unpack_hook):
    y = paddle.mul(a, b)
```
