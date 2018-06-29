# 如何合并模型

由PaddlePaddle训练得到的模型，通常包含两个部分：模型配置文件和参数文件。PaddlePaddle提供工具，将配置文件和参数文件合并成一个文件，即这里所说的**合并的模型**文件，方便在移动端上的离线推断应用中使用。

针对PaddlePaddle的v1和v2 api，我们分别提供了两套合并模型的工具，下面将分别介绍。

## merge\_v2\_model

这个工具适用于使用v2 api训练的模型。`merge_v2_model`是PaddlePaddle提供的一个python函数，使用该工具，首先你需要安装PaddlePaddl的python包。我们以移动端上常用的`Mobilenet`为例，来介绍这个工具的使用。

- **Step 1，准备工作。**
  - 准备**模型配置文件：** 用于推断任务的模型配置文件，必须只包含`inference`网络，即不能包含训练网络中需要的`label`、`loss`以及`evaluator`层。我们使用的基于`Mobilenet`的图像分类任务配置文件见[mobilenet.py](../../../models/standard_network/mobilenet.py)。

  - 准备**参数文件：** 使用PaddlePaddle v2 api训练得到的参数将会存储成`.tar.gz`文件，可直接用于合并模型。我们提供一个使用[flowers102](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/)数据集训练`Mobilenet`分类模型的参数文件[mobilenet_flowers102.tar.gz](http://cloud.dlnel.org/filepub/?uuid=4a3fcd7a-719c-479f-96e1-28a4c3f2195e)。用户可点击参数文件名字通过浏览器下载，或者使用以下命令下载：

  ```bash
  wget -C http://cloud.dlnel.org/filepub/?uuid=4a3fcd7a-719c-479f-96e1-28a4c3f2195e -O mobilenet_flowers102.tar.gz
  ```

- **Step 2，合并模型。**

  运行python脚本[merge_model.py](./merge_model.py)，即可得到合并的模型`mobilenet_flowers102.paddle`。

  ```bash
  $ cat merge_model.py
  import paddle.v2 as paddle
  from paddle.utils.merge_model import merge_v2_model

  # import network configuration
  from mobilenet import mobile_net

  if __name__ == "__main__":
      image_size = 224
      num_classes = 102
      net = mobile_net(3 * image_size * image_size, num_classes, 1.0)
      param_file = './mobilenet_flowers102.tar.gz'
      output_file = './mobilenet_flowers102.paddle'
      merge_v2_model(net, param_file, output_file)
  ```

## paddle\_merge\_model

这个工具适用于使用v1 api训练的模型。`paddle_merge_model`是PaddlePaddle提供的一个可执行文件。假设PaddlePaddle的安装目录位于`PADDLE_ROOT`，该工具的使用方法如下：

```bash
$PADDLE_ROOT/opt/paddle/bin/paddle_merge_model \
    --model_dir="pass-00000" \
    --config_file="config.py" \
    --model_file="output.paddle"
```

该工具需要三个参数：

- `--model_dir`，参数文件所在目录。
- `--config_file`，`inference`网络配置文件的路径。
- `--model_file`，生成的**合并的模型**文件的路径。
