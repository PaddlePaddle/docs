## [仅 API 调用方式不一致]torchvision.transforms.functional.erase

### [torchvision.transforms.functional.erase](https://pytorch.org/vision/main/generated/torchvision.transforms.functional.erase.html?highlight=erase#torchvision.transforms.functional.erase)

```python
torchvision.transforms.functional.erase(
    img: Tensor,
    i: int,
    j: int,
    h: int,
    w: int,
    v: Tensor,
    inplace: bool = False
)
```

### [paddle.vision.transforms.erase](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/vision/transforms/erase_cn.html)

```python
paddle.vision.transforms.erase(
    img: Union[paddle.Tensor, np.ndarray, PIL.Image.Image],
    i: int,
    j: int,
    h: int,
    w: int,
    v: Union[paddle.Tensor, np.ndarray],
    inplace: bool = False
)
```

两者功能一致，参数完全一致，具体如下：
