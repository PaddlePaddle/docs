## [ torch 参数更多 ]torchvision.models.Inception3
### [torchvision.models.Inception3](https://pytorch.org/vision/stable/models/generated/torchvision.models.inception_v3.html)
```python
torchvision.models.Inception3(num_classes: int = 1000, aux_logits: bool = True, transform_input: bool = False, inception_blocks: Optional[list[Callable[..., nn.Module]]] = None, init_weights: Optional[bool] = None, dropout: float = 0.5)
```

### [paddle.vision.models.InceptionV3](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/vision/models/InceptionV3_cn.html)
```python
paddle.vision.models.InceptionV3(num_classes: int = 1000, with_pool: bool = True)
```

torchvision 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| torchvision          | PaddlePaddle | 备注                                                                 |
| ---------------- | ------------ | -------------------------------------------------------------------- |
| num_classes      | num_classes  | 表示分类的类别数。参数完全一致。                                                   |
| aux_logits       | -            | 是否使用辅助分类器，Paddle 无此参数，暂无转写方式。                                |
| transform_input  | -            | 是否对输入进行预处理，Paddle 无此参数，暂无转写方式。                              |
| inception_blocks | -            | 用于构建网络的 Inception 模块，Paddle 无此参数，暂无转写方式。                     |
| init_weights     | -            | 是否对权重进行初始化，Paddle 无此参数，暂无转写方式。                              |
| dropout          | -            | Dropout 概率，Paddle 无此参数，暂无转写方式。                                      |
| -                | with_pool    | 是否在最后的全连接层前使用池化，Paddle 特有参数，使用默认值。                    |
