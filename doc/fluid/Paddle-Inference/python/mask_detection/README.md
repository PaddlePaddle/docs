## 口罩检测


在整个口罩检测任务中，我们会用到两个模型，一个是人脸检测模型，用来检测出图片中的所有的人脸；另外一个为人脸口罩分类模型，用来对人脸进行分类，判别该人脸是否戴有口罩。

在本目录中，我们通过Paddle Inference Python 接口实现了口罩检测任务。

### 运行


**1） 下载模型**

我们有两种方式下载模型：

a. 通过脚本下载

```
cd models
sh model_downloads.sh
```

b. 通过PaddleHub下载

```
# 下载paddlehub以后，通过python执行以下代码
import paddlehub as hub
pyramidbox_lite_mobile_mask = hub.Module(name="pyramidbox_lite_mobile_mask")
# 将模型保存在models文件夹之中
pyramidbox_lite_mobile_mask.processor.save_inference_model(dirname="models")
# 通过以上命令，可以获得人脸检测和口罩佩戴判断模型，分别存储在pyramidbox_lite和mask_detector之中。文件夹中的__model__是模型结构文件，__param__文件是权重文件。
```


**2） 运行程序**

```
python cam_video.py
```

运行后，程序会启动机器上的摄像头并执行口罩检测流程，如果检测到有人脸不带口罩，程序会对该人脸进行红框标记，并显示到屏幕。


![图片1](https://user-images.githubusercontent.com/5595332/81150234-266f4b00-8fb2-11ea-98e7-92909d9c6792.png)
