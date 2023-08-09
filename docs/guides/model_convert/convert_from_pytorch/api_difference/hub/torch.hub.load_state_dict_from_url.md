## [ torch 参数更多 ] torch.hub.load_state_dict_from_url

### [torch.hub.load_state_dict_from_url](https://pytorch.org/docs/2.0/hub.html?highlight=torch+hub+load_state_dict_from_url#torch.hub.load_state_dict_from_url)

```python
torch.hub.load_state_dict_from_url(url,
                                   model_dir=None,
                                   map_location=None,
                                   progress=True,
                                   check_hash=False,
                                   file_name=None)
```

### [paddle.utils.download.get_weights_path_from_url](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/utils/download/get_weights_path_from_url_cn.html)

```python
paddle.utils.download.get_weights_path_from_url(url,
                                                md5sum=None)
```

其中，PyTorch 相比 Paddle 支持更多其他参数，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
|url            |url           |下载的链接。|
|model_dir            |-             |指定文件保存的绝对路径，例如：/tmp/temporary_file，Paddle 无此参数，暂无转写方式|
|map_location    |-             |指定如何重新映射存储位置的函数或 dict，默认为 None，Paddle 无此参数，暂无转写方式。|
|progress       |-             |是否显示进度条，Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。|
|check_hash       |-             |指定下载的 SHA256 文件的前缀，默认为 None，Paddle 无此参数，暂无转写方式。|
|file_name       |-             |下载文件的名称。如果未设置，将使用 url 中的文件名，默认为 None，Paddle 无此参数，暂无转写方式。|
|-              |md5sum        |下载文件的 md5 值。Pytorch 无此参数，Paddle 保持默认即可。|
