## [torch 参数更多]torchvision.io.decode_jpeg

### [torchvision.io.decode_jpeg](https://pytorch.org/vision/main/generated/torchvision.io.decode_jpeg.html)

```python
torchvision.io.decode_jpeg(
    input: Union[Tensor, List[Tensor]],
    mode: ImageReadMode = ImageReadMode.UNCHANGED,
    device: Union[str, torch.device] = 'cpu',
    apply_exif_orientation: bool = False
) -> Union[Tensor, List[Tensor]]
```

### [paddle.vision.ops.decode_jpeg](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/vision/ops/decode_jpeg_cn.html)

```python
paddle.vision.ops.decode_jpeg(
    x: paddle.Tensor,
    mode: str = 'unchanged'
) -> paddle.Tensor
```

两者功能一致，但 PyTorch 相比 Paddle 支持更多其他参数，在 Paddle 中不需要指定 `device` 和 `apply_exif_orientation` 参数，它们在代码转写时可以直接删除，因为它们对主要功能没有直接影响。

### 参数映射

| torchvision                           | PaddlePaddle       | 备注      |
| ------------------------------------- | ------------------ | -------- |
| input                                 | x                  | 输入 JPEG 图像的原始字节，Paddle 仅支持一个 Tensor，PyTorch 支持单个或列表。|
| mode                                  | mode               | 图像模式选择，e.g. "RGB"，参数含义一致。|
| device                                | -                  | 指定解码后图像存储的设备，Paddle 无此参数，通常仅影响图像存储位置，可直接删除。|
| apply_exif_orientation                | -                  | 是否应用 EXIF 方向变换，Paddle 无此参数，可直接删除。|

### 转写示例

```python
# PyTorch 写法
torchvision.io.decode_jpeg(input=image_bytes, mode='RGB')

# Paddle 写法
paddle.vision.ops.decode_jpeg(x=image_bytes, mode='RGB')
```
