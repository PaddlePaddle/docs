## [paddle 参数更多]torchvision.transforms.Pad

### [torchvision.transforms.Pad](https://pytorch.org/vision/main/generated/torchvision.transforms.Pad.html)

```python
torchvision.transforms.Pad(padding: int | list | tuple, fill: int | list | tuple = 0, padding_mode: str = 'constant')
```

### [paddle.vision.transforms.Pad](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/vision/transforms/Pad__upper_cn.html#pad)

```python
paddle.vision.transforms.Pad(padding: int | list | tuple, fill: int | list | tuple = 0, padding_mode: str = 'constant', keys: list[str] | tuple[str] = None)
```

两者功能一致，但 Paddle 相比 torchvision 支持更多参数，具体如下：

### 参数映射

| torchvision                   | paddle                       | 备注                                                         |
| --------------------------------------------- | ---------------------------------------------------- | ------------------------------------------------------------ |
| padding (int \| list \| tuple)                     | padding (int \| list \| tuple)                       | 两者均支持单个整数或序列进行填充。                           |
| fill (int \| list \| tuple)                  | fill (int \| list \| tuple)                          | Paddle 支持列表或元组，用于多通道图像填充。                |
| padding_mode (str)                            | padding_mode (str)                                   | 两者均支持 'constant', 'edge', 'reflect', 'symmetric' 模式。|
| -                                             | keys (list[str] \| tuple[str] = None)                | Paddle 支持 `keys` 参数。            |
