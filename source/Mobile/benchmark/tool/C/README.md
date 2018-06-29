# Inference demo

This is an inference demo program based on the C API of PaddlePaddle.
The demo explained here is based on the C++ code, so we need to use g++ or clang++ to compile.
The demo can be run from the command line and can be used to test the inference performance of various different models.

## Android
To compile and run this demo in an Android environment, please follow the following steps:

- **Step 1, build PaddlePaddle for Android.**

  Refer to [this document](https://github.com/PaddlePaddle/Paddle/blob/develop/doc/mobile/cross_compiling_for_android_en.md) to compile the Android version of PaddlePaddle. After following the mentioned steps, make install will generate an output directory containing three subdirectories: include, lib, and third_party( `libpaddle_capi_shared.so` will be produced in the `lib` directory).

- **Step 2, build the inference demo.**

  Compile `inference.cc` to an executable program for the Android environment as follows:

    - For armeabi-v7a

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

    - For arm64-v8a

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

- **Step 3, prepare a merged model.**

  Models config(.py) (eg: [Mobilenet](https://github.com/PaddlePaddle/Mobile/blob/develop/models/mobilenet.py)) contain only the structure of our models. A developer can choose [model config here](https://github.com/PaddlePaddle/Mobile/tree/develop/models) to train their custom models. PaddlePaddle documentation has [several tutorials](https://github.com/PaddlePaddle/models) for building and training models. The model parameter file(.tar.gz) will be generated during the training process. There we need to merge the configuration file(.py) and the parameter file(.tar.gz) into a file. Please refer to the [details.](https://github.com/PaddlePaddle/Mobile/tree/develop/tools/merge_config_parameters)

- **Step 4, run the demo.**

  Users can run the demo program by logging into the Android environment via [adb](https://developer.android.google.cn/studio/command-line/adb.html?hl=zh-cn#Enabling) and specifying the PaddlePaddle model from the command line as follows:

    ```bash
    $ adb push inference /data/local/tmp # transfer the executable to Android's memory
    $ adb push mobilenet_flowers102.paddle /data/local/tmp # transfer the model to Android's memory
    $ adb shell # login Android device
    odin:/ $ cd /data/local/tmp # switch to the working directory
    odin:/data/local/tmp $ ls
    inference  mobilenet_flowers102.paddle
    odin:/data/local/tmp $ chmod +x inference
    odin:/data/local/tmp $ ./inference --merged_model ./mobilenet_flowers102.paddle --input_size 150528 # run the executable
    I1211 17:12:53.334666  4858 Util.cpp:166] commandline:
    Time of init paddle 3.4388 ms.
    Time of create from merged model file 141.045 ms.
    Time of forward time 398.818 ms.
    ```

    **Note:** `input_size` is 150528, cause that the input size of the model is `3 * 224 * 224 = 150528`
