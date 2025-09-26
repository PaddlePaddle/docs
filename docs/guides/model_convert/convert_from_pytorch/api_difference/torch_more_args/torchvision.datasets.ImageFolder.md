## [torch 参数更多]torchvision.datasets.ImageFolder

### [torchvision.datasets.ImageFolder](https://pytorch.org/vision/main/generated/torchvision.datasets.ImageFolder.html)

```python
torchvision.datasets.ImageFolder(root: Union[str, pathlib.Path],
                                 transform: Optional[Callable] = None,
                                 target_transform: Optional[Callable] = None,
                                 loader: Callable[[str], Any] = default_loader,
                                 is_valid_file: Optional[Callable[[str], bool]] = None,
                                 allow_empty: bool = False)
```

### [paddle.vision.datasets.ImageFolder](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/vision/datasets/ImageFolder_cn.html)

```python
paddle.vision.datasets.ImageFolder(root: str,
                                   loader: Optional[Callable] = None,
                                   extensions: Optional[Union[list[str], tuple[str]]] = None,
                                   transform: Optional[Callable] = None,
                                   is_valid_file: Optional[Callable[[str], bool]] = None)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| torchvision | PaddlePaddle | 备注                                      |
| -------------------------------- | ---------------------------------- | ----------------------------------------- |
| root                             | root                               | 根目录路径。                                |
| transform                        | transform                          | 图片数据的预处理。 |
| target_transform                 | -                                  | 目标标签的预处理，Paddle 无此参数，暂无转写方式。         |
| loader                           | loader                             | 图片加载函数。                              |
| is_valid_file                    | is_valid_file                      | 根据数据路径判断是否合法。               |
| allow_empty                      | -                                  | 是否允许空文件夹，Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。 |
| -                                | extensions                         | 限制数据集文件的格式，PyTorch 无此参数，Paddle 保持默认即可。 |
