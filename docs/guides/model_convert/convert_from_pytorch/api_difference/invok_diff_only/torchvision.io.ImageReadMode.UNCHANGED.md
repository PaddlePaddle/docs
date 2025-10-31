## [ 仅 API 调用方式不一致 ]torchvision.io.ImageReadMode.UNCHANGED

### [torchvision.io.ImageReadMode.UNCHANGED](https://pytorch.org/vision/stable/generated/torchvision.io.ImageReadMode.html#torchvision.io.ImageReadMode.UNCHANGED)

```python
torchvision.io.ImageReadMode.UNCHANGED
```

两者功能一致，但调用方式不一致，具体如下：

### 转写示例

```python
## PyTorch 写法
mode = torchvision.io.ImageReadMode.UNCHANGED

## Paddle 写法
mode = 'unchanged'
```
