## [输入参数类型不一致]torchvision.io.decode_jpeg

### [torchvision.io.decode_jpeg](https://pytorch.org/vision/main/generated/torchvision.io.decode_jpeg.html)

```python
torchvision.io.decode_jpeg(input: Union[Tensor, List[Tensor]], mode: ImageReadMode = ImageReadMode.UNCHANGED, device: Union[str, device] = 'cpu', apply_exif_orientation: bool = False)
```

### [paddle.vision.ops.decode_jpeg](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/vision/ops/decode_jpeg_cn.html)

```python
paddle.vision.ops.decode_jpeg(x, mode='unchanged', name=None)
```

两者功能一致，但输入参数类型不一致，具体如下：

### 参数映射

| torchvision                           | PaddlePaddle       | 备注      |
| ------------------------------------- | ------------------ | -------- |
| input                                 | x                  | 包含 JPEG 图像原始字节，仅参数名不一致。 |
| mode                                  | mode               | 转换图像模式选择，PyTorch 参数为 string 或 ImageReadMode 枚举类, Paddle 参数为 string，需要转写。 |
| device                                | -                  | 解码后的图像将被存储到的设备，Paddle 无此参数，需要转写。 |
| apply_exif_orientation                | -                  | 对输出张量应用 EXIF 方向变换，Paddle 无此参数，暂无转写方式。 |

### 转写示例

#### mode：转换图像模式选择
```python
# PyTorch 写法
torchvision.io.decode_jpeg(input=image_bytes, mode=torchvision.io.ImageReadMode.RGB)

# Paddle 写法
paddle.vision.ops.decode_jpeg(x=image_bytes, mode='RGB')
```

### device：解码后的图像将被存储到的设备

```python
# PyTorch 写法
y = torchvision.io.decode_jpeg(input=image_bytes, device=torch.device('cpu'))

# Paddle 写法
y = paddle.vision.ops.decode_jpeg(x=image_bytes)
y.cpu()
```
