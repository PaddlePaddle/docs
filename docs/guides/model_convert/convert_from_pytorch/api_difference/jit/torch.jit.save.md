## [torch 参数更多] torch.jit.save

### [torch.jit.save](https://pytorch.org/docs/1.13/generated/torch.jit.save.html?highlight=save#torch.jit.save)

```python
torch.jit.save(m,
                f,
                _extra_files=None)
```

### [paddle.jit.save](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/jit/save_cn.html)

```python
paddle.jit.save(layer,
                path,
                input_spec=None,
                **configs)
```

两者功能一致但参数不一致，部分参数名不同，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
|m              |layer         | 需要存储的 Layer 对象或者 function，参数名不同。       |
|f              |-          |一个类似文件的对象（必须实现 write 和 flush 方法）或包含文件名的字符串，如果是文件对象，Paddle 暂无转写方式；如果是文件名，则可以用 path 参数实现,需要进行转写。|
|_extra_files   |-             |文件名到内容的映射，这些内容将作为 f 的一部分进行存储，对应 Paddle 中的 input_spec 参数，需要进行转写。|
|-              |path         |存储模型的路径前缀，格式为 dirname/file_prefix 或者 file_prefix,对应 PyTorch 的 f 参数。需要进行转写。|
|-              |input_spec    |描述存储模型 forward 方法的输入，可以通过 InputSpec 或者示例 Tensor 进行描述，对应 PyTorch 中的_extra_files 参数，需要进行转写。|

### 转写示例
#### f：一个文件名
```python
# Pytorch 写法
torch.jit.save(m, f=filePath)

# Paddle 写法
paddle.jit.save(fun, path=filePath)
```
#### _extra_files: 文件名到内容的映射，这些内容将作为 f 的一部分进行存储
```python
# Pytorch 写法
torch.jit.save(m, f=filePath, _extra_files={'foo.txt': b'bar'})

# Paddle 写法
paddle.jit.save(fun, path=filePath,  input_spec=list[Tensor])
```
