# 示例程序

这是一个用C++语言编写的简单的示例程序，其中通过调用Paddle的C-API接口，实现对模型的推断。

这个示例程序可在linux或Android手机的命令行下运行，可以用来测试不同模型的性能。

## Android
用户可按照以下几个步骤，编译出在Android设备上执行的可执行程序。

- **Step 1，编译Android平台上适用的PaddlePaddle库。**

    用户可以按照[Android平台编译指南](https://github.com/PaddlePaddle/Paddle/blob/develop/doc/mobile/cross_compiling_for_android_cn.md)，拉取PaddlePaddle最新代码，编译Android平台上适用的PaddlePaddle库。在执行`make install`之后，PaddlePaddle库将会安装在`CMAKE_INSTALL_PREFIX`所指定的目录下。该目录包含如下几个子目录：
    - `include`，其中包含使用PaddlePaddle所需要引入的头文件，通常代码中加入`#include <paddle/capi.h>`即可。
    - `lib`，其中包含了PaddlePaddle对应架构的库文件。其中包括：
      - 动态库，`libpaddle_capi_shared.so`。
      - 静态库，`libpaddle_capi_layers.a`和`libpaddle_capi_engine.a`。
    - `third_party`，PaddlePaddle所依赖的第三方库。

    你也可以从[wiki](https://github.com/PaddlePaddle/Mobile/wiki)下载编译好的版本。

- **Step 2，编译示例程序。**

    示例程序项目使用CMake管理，可按照以下步骤，编译Android设备上运行的可执行程序。
     这个步骤中依旧需要用到第一步中配置的**独立工具链**。

    - armeabi-v7a架构

    ```bash
    $ git clone https://github.com/PaddlePaddle/Mobile.git
    $ cd Mobile/benchmark/tool/C/
    $ mkdir build
    $ cd build
    $ cmake .. \
            -DANDROID_ABI=armeabi-v7a \
            -DANDROID_STANDALONE_TOOLCHAIN=your/path/to/arm_standalone_toolchain \
            -DPADDLE_ROOT=The output path generated in the first step \
            -DCMAKE_BUILD_TYPE=MinSizeRel

    $ make
    ```

    - arm64-v8a架构

    ```bash
    $ git clone https://github.com/PaddlePaddle/Mobile.git
    $ cd Mobile/benchmark/tool/C/
    $ mkdir build
    $ cd build

    $ cmake .. \
            -DANDROID_ABI=arm64-v8a \
            -DANDROID_STANDALONE_TOOLCHAIN=your/path/to/arm64_standalone_toolchain \
            -DPADDLE_ROOT=The output path generated in the first step \
            -DCMAKE_BUILD_TYPE=MinSizeRel

    $ make
    ```

    执行上述命令执行，会在`build`目录下生成目标可执行文件`inference`。

- **Step 3，准备模型。**

    Android设备上推荐使用**合并的模型（merged model）**。以Mobilenet为例，要生成**合并的模型**文件，首先你需要准备以下文件：
    - 模型配置文件[mobilenet.py](https://github.com/PaddlePaddle/Mobile/tree/develop/models/standard_network/mobilenet.py)，它是使用PaddlePaddle的v2 api编写的`Mobilenet`模型的网络结构。当前repo的[models](https://github.com/PaddlePaddle/Mobile/tree/develop/models)目录下维护了一些移动端常用的PaddlePaddle网络配置。同时，用户可在[models](https://github.com/PaddlePaddle/models)repo下面找到更多PaddlePaddle常用的网络配置，该repo下面同时提供了使用PaddlePaddle训练模型的方法。
    - 模型参数文件。使用PaddlePaddle v2 api训练得到的参数将会存储成`.tar.gz`文件。比如，我们提供了一个使用[flowers102](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/)数据集训练`Mobilnet`分类模型的参数文件[mobilenet\_flowers102.tar.gz](http://cloud.dlnel.org/filepub/?uuid=4a3fcd7a-719c-479f-96e1-28a4c3f2195e)。你也可以使用以下命令下载该参数文件：

    ```bash
    wget -C http://cloud.dlnel.org/filepub/?uuid=4a3fcd7a-719c-479f-96e1-28a4c3f2195e -O mobilenet_flowers102.tar.gz
    ```

    **注意，用来`merge model`的模型配置文件必须只包含`inference`网络**。

    在准备好模型配置文件（.py）和参数文件（.tar.gz）之后，且所在机器已经成功安装了PaddlePaddle的Python包之后，我们可以通过执行以下脚本生成需要的`merged model`。

    ```bash
    $ cd Mobile/deployment/model/merge_config_parameters
    $ python merge_model.py
    ```

    你也可以直接下载[mobilenet\_flowers102.paddle](http://cloud.dlnel.org/filepub/?uuid=d3b95cf9-4dc3-476f-bdc7-98ac410c4f71)试用。命令行下载方式：

    ```
    wget -C http://cloud.dlnel.org/filepub/?uuid=d3b95cf9-4dc3-476f-bdc7-98ac410c4f71 -O mobilenet_flowers102.paddle
    ```

    更多有关于生成`merged model`的详情，请参考[merge\_config\_parameters](https://github.com/PaddlePaddle/Mobile/tree/develop/deployment/model/merge_config_parameters/README.cn.md)。

- **Step 4，在Android设备上测试。**

     这是一个可以在Android设备上运行的命令行测试程序。你可以通过桌面终端，借助于[adb](https://developer.android.google.cn/studio/command-line/adb.html?hl=zh-cn#Enabling)工具，传输数据到Android设备上，并且登陆Android设备，运行可执行程序进行测试。

    ```bash
    $ adb push inference /data/local/tmp # 将可执行程序传输到Android设备上
    $ adb push mobilenet_flowers102.paddle /data/local/tmp # 将模型文件传输到Android设备上
    $ adb shell # 登陆Android设备
    odin:/ $ cd /data/local/tmp # 进入工作目录
    odin:/data/local/tmp $ ls
    inference  mobilenet_flowers102.paddle
    odin:/data/local/tmp $ chmod +x inference
    odin:/data/local/tmp $ ./inference --merged_model ./mobilenet_flowers102.paddle --input_size 150528 # 执行测试程序
    I1211 17:12:53.334666  4858 Util.cpp:166] commandline:
    Time of init paddle 3.4388 ms.
    Time of create from merged model file 141.045 ms.
    Time of forward time 398.818 ms.
    ```

    `inference`可执行程序需要设置两个运行时参数：
    - `--merged_model`，模型的路径。
    - `--input_size`，模型输入数据的长度。由于`mobilenet`使用`3 x 224 x 224`图像数据作为输入，因此设置`--input_size 150528`。

## 注意

该示例程序只是用来简单地测试Android设备上，模型的推断速度。因为使用随机数据作为模型的输入，若要测试和验证模型的正确性，请根据实际的需求进行修改。
