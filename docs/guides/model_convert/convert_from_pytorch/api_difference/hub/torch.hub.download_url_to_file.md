## [torch 参数更多] torch.hub.download_url_to_file

### [torch.hub.download_url_to_file](https://pytorch.org/docs/1.13/hub.html?highlight=download#torch.hub.download_url_to_file)

```python
torch.hub.download_url_to_file(url,
                                dst,
                                hash_prefix=None,
                                progress=True)
```

### [paddle.hub.download_url_to_file](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/utils/download/get_weights_path_from_url_cn.html)

```python
paddle.utils.download.get_weights_path_from_url(url,
                                                md5sum=None)
```

两者功能一致但参数不一致，部分参数名不同，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
|url            |url           |下载的链接。|
|dst            |-             |对象将被保存的完整路径，例如：/tmp/temporary_file，Paddle 无此参数。|
|hash_prefix    |-             |如果不为 None，则下载的 SHA256 文件应以 hash_prefix 开头，默认为 None，Paddle 无此参数。|
|progress       |-             |是否显示进度条。默认值为 True，Paddle 无此参数，可直接删除。|
|-              |md5sum        | 下载文件的 md5 值，默认值：None。|
