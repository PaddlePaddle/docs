## [torch 参数更多 ]torch.load
### [torch.load](https://pytorch.org/docs/stable/generated/torch.load.html?highlight=load#torch.load)

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

其中 Pytorch 相比 Paddle 支持更多其他参数，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| f             | path         | 载入目标对象实例的路径/内存对象， 仅参数名不一致。                   |
| map_location  | -            | 表示加载模型的位置，PaddlePaddle 无此参数。                   |
| pickle_module | -            | 表示用于 unpickling 元数据和对象的模块，PaddlePaddle 无此参数。                       |
| weights_only  | -            | 指示 unpickler 是否应限制为仅加载张量、基元类型和字典，PaddlePaddle 无此参数。                   |
| pickle_load_args| -          | 传递给 pickle_module.load（）和 pickle_mdule.Unpickler（）的可选关键字参数，PaddlePaddle 无此参数。                   |


### 转写示例
四个 torch 多支持的参数（map_location，pickle_modeule，weights_only，pickle_load_args），Paddle 暂无转写方式
