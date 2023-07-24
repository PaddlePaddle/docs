## [ torch 参数更多 ] torch.hub.download_url_to_file

### [torch.hub.download_url_to_file](https://pytorch.org/docs/stable/hub.html?highlight=download#torch.hub.download_url_to_file)

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

其中，PyTorch 相比 Paddle 支持更多其他参数，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
|url            |url           |下载的链接。|
|dst            |-             |指定文件保存的绝对路径，例如：/tmp/temporary_file，Paddle 无此参数，暂无转写方式|
|hash_prefix    |-             |指定下载的 SHA256 文件的前缀，默认为 None，Paddle 无此参数，暂无转写方式。|
|progress       |-             |是否显示进度条，Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。|
|-              |md5sum        |下载文件的 md5 值。Pytorch 无此参数，Paddle 保持默认即可。|
