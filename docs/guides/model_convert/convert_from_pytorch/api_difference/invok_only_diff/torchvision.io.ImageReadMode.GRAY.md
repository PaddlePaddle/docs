## [ 仅 API 调用方式不一致 ]torchvision.io.ImageReadMode.GRAY

### [torchvision.io.ImageReadMode.GRAY](https://pytorch.org/vision/stable/generated/torchvision.io.ImageReadMode.html#torchvision.io.ImageReadMode.GRAY)

```python
torchvision.io.ImageReadMode.GRAY
```

两者功能一致，但调用方式不一致，具体如下：

### 转写示例

```python
## PyTorch 写法
mode = torchvision.io.ImageReadMode.GRAY

## Paddle 写法
mode = 'gray'
```
