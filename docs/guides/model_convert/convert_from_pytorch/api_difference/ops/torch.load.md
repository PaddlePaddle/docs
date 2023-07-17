## [torch 参数更多 ]torch.load

### [torch.load](https://pytorch.org/docs/1.13/generated/torch.load.html?highlight=load#torch.load)

```python
torch.load(f,
           map_location=None,
           pickle_module=pickle,
           *,
           weights_only=False,
           **pickle_load_args)
```

### [paddle.load](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/load_cn.html#load)

```python
paddle.load(path,
            **configs)
```

Pytorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch          | PaddlePaddle | 备注                                                         |
| ---------------- | ------------ | ------------------------------------------------------------ |
| f                | path         | 载入目标对象实例的路径/内存对象， 仅参数名不一致。           |
| map_location     | -            | 表示如何重新映射存储位置，Paddle 无此参数, 可直接删除。               |
| pickle_module    | -            | 表示用于 unpickling 元数据和对象的模块，Paddle 无此参数, 可直接删除。 |
| weights_only     | -            | 指示 unpickler 是否应限制为仅加载张量、原始类型和字典，Paddle 无此参数, 可直接删除。 |
| pickle_load_args | -            | 传递给 pickle_module.load（）和 pickle_mdule.Unpickler（）的可选关键字参数，Paddle 无此参数, 可直接删除。 |
| -                | configs      | 表示其他用于兼容的载入配置选项。PyTorch 无此参数， Paddle 保持默认即可。 |
