## [torch 参数更多 ]torch.save
### [torch.save](https://pytorch.org/docs/stable/generated/torch.save.html?highlight=save#torch.save)

```python
torch.save(obj,
           f,
           pickle_module=pickle,
           pickle_protocol=DEFAULT_PROTOCOL,
           _use_new_zipfile_serialization=True)
```

### [paddle.save](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/save_cn.html#save)

```python
paddle.save(obj,
            path,
            protocol=4)
```

其中 Pytorch 相比 Paddle 支持更多其他参数，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| obj           | obj          | 要保存的对象实例，torch 支持 io.BytesIO、io.StringIO、文件，paddle 只支持文件。|
| f             | path         | 表示存储的路径。                   |
| pickle_module | -            | 表示用于 pickling 元数据和对象的模块，PaddlePaddle 无此参数, 可直接删除。 |
| pickle_protocol| protocol    | pickle 模块的协议版本。                   |
| _use_new_zipfile_serialization | -            | 是否以旧格式加载文件，PaddlePaddle 无此参数, 可直接删除。 |


### 转写示例
四个 torch 多支持的参数（pickle_modeule，obj，f，_use_new_zipfile_serialization），Paddle 暂无转写方式
