## [paddle 参数更多]torchvision.transforms.functional.pad

### [torchvision.transforms.functional.pad](https://pytorch.org/vision/main/generated/torchvision.transforms.functional.pad.html)

```python
torchvision.transforms.functional.pad(
    img: Union[PIL.Image.Image, torch.Tensor],
    padding: Union[int, List[int], Tuple[int, ...]],
    fill: Union[int, List[int], Tuple[int, ...]] = 0,
    padding_mode: Literal['constant', 'edge', 'reflect', 'symmetric'] = 'constant'
)
```

### [paddle.vision.transforms.pad](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/vision/transforms/pad_cn.html)

```python
paddle.vision.transforms.pad(
    img: Union[PIL.Image.Image, np.ndarray, paddle.Tensor],
    padding: Union[int, List[int], Tuple[int, ...]],
    fill: Union[int, List[int], Tuple[int, ...]] = 0,
    padding_mode: Literal['constant', 'edge', 'reflect', 'symmetric'] = 'constant',
    keys: Optional[Union[List[str], Tuple[str, ...]]] = None
)
```

两者功能一致，但 Paddle 相比 torchvision 支持更多参数，具体如下：

### 参数映射

| torchvision                   | PaddlePaddle| 备注                                                         |
| --------------------------------------------- | ---------------------------------------------------- | ------------------------------------------------------------ |
| img  | img   | 被填充的图像。|
| padding                      | padding                        | 两者均支持单个整数或序列进行填充。                           |
| fill                   | fill                           | 用于多通道图像填充。                |
| padding_mode                             | padding_mode                                    | 两者均支持 'constant', 'edge', 'reflect', 'symmetric' 模式。|
| -                                             | keys                 | Paddle 支持 `keys` 参数，PyTorch 无此参数，Paddle 保持默认即可。            |
