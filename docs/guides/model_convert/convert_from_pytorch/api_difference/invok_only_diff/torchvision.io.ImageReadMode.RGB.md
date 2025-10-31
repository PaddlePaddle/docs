## [ 仅 API 调用方式不一致 ]torchvision.io.ImageReadMode.RGB

### [torchvision.io.ImageReadMode.RGB](https://pytorch.org/vision/stable/generated/torchvision.io.ImageReadMode.html#torchvision.io.ImageReadMode.RGB)

```python
torchvision.io.ImageReadMode.RGB
```

两者功能一致，但调用方式不一致，具体如下：

### 转写示例

```python
## PyTorch 写法
mode = torchvision.io.ImageReadMode.RGB

## Paddle 写法
mode = 'rgb'
```
