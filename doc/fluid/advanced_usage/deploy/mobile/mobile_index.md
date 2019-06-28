# Paddle-Mobile

## 简介

## 使用方法

目前有两种 C++ 接口可以实现 mobile 预测：

- CxxConfig: 完整功能预测接口
- MobileConfig: 专用于移动端的轻量级接口

对应的 Java 接口也有两种：

- loadCxxModel: 完整功能预测接口
- loadMobileModel: 专用于移动端的轻量级接口

前者输入原始预测模型，并执行相应的计算图优化后，实现高性能预测；后者输入计算图优化之后的模型，直接执行相关计算。

### Java Basics

#### 编译

Java 接口需要在 cmake 选项中同时打开 DWITH_LITE, DLITE_WITH_JAVA, DLITE_WITH_ARM。 例如：

```shell
# ARM_TARGET_OS in "android" , "armlinux"
# ARM_TARGET_ARCH_ABI in "armv8", "armv7" ,"armv7hf"
# ARM_TARGET_LANG in "gcc" "clang"
mkdir -p build.lite.android.arm8.gcc
cd build.lite.android.arm8.gcc

cmake .. \
  -DWITH_GPU=OFF \
  -DWITH_MKL=OFF \
  -DWITH_LITE=ON \
  -DLITE_WITH_JAVA=ON \
  -DLITE_WITH_CUDA=OFF \
  -DLITE_WITH_X86=OFF \
  -DLITE_WITH_ARM=ON \
  -DLITE_WITH_LIGHT_WEIGHT_FRAMEWORK=ON \
  -DWITH_TESTING=ON \
  -DARM_TARGET_OS=android -DARM_TARGET_ARCH_ABI=armv8 -DARM_TARGET_LANG=gcc

make -j4
```

make 成功后，Linux下会生成动态库文件 paddle/fluid/lite/api/android/jni/libpaddle_lite_jni.so（ Mac 下为 
libpaddle_lite_jni.jnilib, Windows 下为libpaddle_lite_jni.dll ）该动态库即 Java JNI ( Java Native Interface ) 所需要的
C++ 接口动态链接库，下面例子中我们将使用 Linux 下 libpaddle_lite_jni.so 为例。同时，也会在同一个文件夹下生成 
PaddlePredictor.jar

#### Android 程序构建

在我们的库中，Java 代码库被放在 paddle/fluid/lite/api/android/jni/src 中，具体有两个classes:

com.baidu.paddle.lite.PaddlePredictor
com.baidu.paddle.lite.Place

你可以将其打包成 .jar 或者直接使用 Java 源代码接口。如果要使用 .jar，我们上节编译中生成的 .jar 也可以直接使用。

请将 JNI 动态链接库放在 Android Studio 代码 jniLibs 文件夹对应的体系结构文件夹下。例如要在 arm8 架构的手机，就 在 src/main/jniLibs/arm8 文件夹下放置 libpaddle_lite_jni.so，文件路径如果不存在请创建。

接下来，我们将具体介绍PaddlePredictor.java 和 Place.java 

#### 代码接口 Place

Paddle 预测中，为了便于管理不同的硬件及kernel 的其他实现细节，定义如下四个信息：

- Target: 具体的硬件空间，比如 `ARM` 表示 ARM CPU，`OPEN_CL` 表示 OpenCL
- DataLayout: Tensor 中的数据排布，目前有 `NCHW`
- Precison: kernel 的计算精度，或者 Tensor 的存储类型，目前有 `FLOAT`, `INT8` 等
- `Device`: 硬件的 device id，可以是 0 开始的整数

前三个为Java enum，最后一个为整型。相关定义如下

```java
public enum TargetType {
  UNKNOWN(0), HOST(1), X86(2), CUDA(3), ARM(4), OPEN_CL(5), ANY(6);
}
public enum PrecisionType {
  UNKNOWN(0), FLOAT(1), INT8(2), INT32(3), ANY(4);
}
public enum DataLayoutType {
  UNKNOWN(0), NCHW(1), ANY(2);
}
```

而 Place 就是这四个信息的整合，其数据结构为

```java
public class Place {
  public TargetType target;
  public PrecisionType precision;
  public DataLayoutType layout;
  public int device;
};
```

Place 用于标记Kernel 的主要计算模式，比如`place.precision=INT8` 的 kernel 表示为 Int8量化的 kernel。Place 暴露给用户，用户帮助指定模型硬件及量化等模式。

#### 代码接口 PaddlePredictor

PaddlePredictor 提供的 methods 都是 native static methods。整体上运行的思路为
载入模型 -> 设置输入 -> 运行模型 -> 获取输出/存储运行后优化的模型 -> 清理掉载入的模型

我们将介绍各个步骤的主要功能，具体接口的参数和返回值请见Javadoc：

1. 载入模型：

	```java
	// 载入没有优化过的原始模型，用户可以设置期望的 Place 和可选的 Place 
	public static native boolean loadCxxModel(String modelPath, Place preferredPlace, Place[] validPlaces); 
	
	// 载入没有优化过的原始模型，用户可以设置期望的 Place 和可选的 Place 
	public static native boolean loadMobileModel(String modelPath);
	```

2. 设置输入

	```java
	// 设置第 offest （从0开始）输入的维度和float数据
	public static native boolean setInput(int offset, int[] dims, float[] buf);
	
	// 设置第 offest （从0开始）输入的维度和byte数据 （在c++端为int8）
	public static native boolean setInput(int offset, int[] dims, byte[] buf);
	```

3. 运行模型
	
	```java
	// 运行模型
	public static native boolean run();
	```

4. 获取输出
	
	```java
	// 获取第 offset （从0开始）的 float 输出
	public static native float[] getFloatOutput(int offset);
	// 获取第 offset （从0开始）的 byte 输出
	public static native byte[] getByteOutput(int offset);
	// 指定名字获取 Var 的 float 输出
	public static native float[] fetchFloat(String name);
	// 指定名字获取 Var 的 byte 输出
	public static native byte[] fetchByte(String name);
	```

5. 存储运行后优化的模型

	```java
	public static native boolean saveOptimizedModel(String modelPath);
	```

6. 清理掉载入的模型
	
	```java
	public static native boolean clear();
	```

使用示例如下：

```java
String modelPath = "lite_naive_model"; // 用户定义的模型路径

// 用户自定义的输入，例子里是 100 * 100 的 float
float[] inputBuffer = new float[10000];
for (int i = 0; i < 10000; ++i) {
inputBuffer[i] = i;
}
int[] dims = {100, 100};

// Cxx Model 设定 Place
Place preferredPlace = new Place(Place.TargetType.X86, Place.PrecisionType.FLOAT);
Place[] validPlaces = new Place[2];
validPlaces[0] = preferredPlace;
validPlaces[1] = new Place(Place.TargetType.ARM, Place.PrecisionType.FLOAT);

// 载入模型
PaddlePredictor.loadCxxModel(modelPath, preferredPlace, validPlaces);
// 设置输入
PaddlePredictor.setInput(0, dims, inputBuffer);
// 运行Predictor
PaddlePredictor.run();
// 获取输出
float[] cxxOutput = PaddlePredictor.getFloatOutput(0);
// 保持优化后的模型在新路径
String optimizedModelPath = modelPath + ".opt";
PaddlePredictor.saveOptimizedModel(optimizedModelPath);
// 清除已载入的模型
PaddlePredictor.clear();

// Mobile Model 载入优化后的模型
PaddlePredictor.loadMobileModel(optimizedModelPath);
// 设置输入
PaddlePredictor.setInput(0, dims, inputBuffer);
// 运行
PaddlePredictor.run();
// 获取输出
float[] mobileOutput = PaddlePredictor.getFloatOutput(0);
```


### C++ Basics

在使用前，有几个基本概念：

#### Place

Place 在 C++ 中概念与 Java 相同，为了便于管理不同的硬件及kernel 的其他实现细节，定义如下四个信息：

- Target: 具体的硬件空间，比如 `kARM` 表示 ARM CPU，`kOpenCL` 表示 OpenCL
- DataLayout: Tensor 中的数据排布，目前有 `kNCHW`
- Precison: kernel 的计算精度，或者 Tensor 的存储类型，目前有 `kFloat`, `kInt8` 等
- `Device`: 硬件的 device id，可以是0开始的整数

前三个为结构体，最后一个为整型。相关定义如下

```c++
enum class TargetType : int {
  kUnk = 0,
  kHost,
  kX86,
  kCUDA,
  kARM,
  kOpenCL,
  kAny,  // any target
  NUM,   // number of fields.
};
enum class PrecisionType : int {
  kUnk = 0,
  kFloat,
  kInt8,
  kInt32,
  kAny,  // any precision
  NUM,   // number of fields.
};
enum class DataLayoutType : int {
  kUnk = 0,
  kNCHW,
  kAny,  // any data layout
  NUM,   // number of fields.
};
```

而 Place 就是这四个信息的整合，其数据结构为

```c++
struct Place {
  TargetType target{TARGET(kUnk)};
  PrecisionType precision{PRECISION(kUnk)};
  DataLayoutType layout{DATALAYOUT(kUnk)};
  int16_t device{0};  // device ID
};
```

Place 用于标记Kernel 的主要计算模式，比如`place.precision=kInt8` 的 kernel 表示为 Int8量化的 kernel。Place 暴露给用户层，用户帮助指定模型执行的硬件及量化等执行模式。

#### Config

预测接口使用的第一步是执行 `CreatePaddlePredictor(config)` 接口创建一个 predictor，具体的 config 目前有多个选择，对应着也会模板特化出不同的 predictor以适应不同的场景。

模板接口如下

```c++
template <typename ConfigT>
std::shared_ptr<PaddlePredictor> CreatePaddlePredictor(const ConfigT&);
```

接下来会详细介绍两种 Config: `CxxConfig` 和 `MobileConfig`.

### CxxConfig 及对应 Predictor

接口如下：

- `set_model_dir(const std::string& x)` 设置模型路径(目前只支持 `__model__` + `params` 两个文件的模型格式)
- `set_preferred_place(const Place& x)` 设置期望的执行 Place
- `set_valid_places(const std::vector<Place>& x)`设置可选的 Place

`valid_places` 用于设置模型可执行的 Place 范围，底层会根据place 信息挑选出具体的硬件执行 kernel，而`preferred_place` 用于指定 `valid_places` 中最优先执行的 Place，从而使对应 place 的 kernel 更优先被选择.

比如，要执行 ARM FP32 量化预测，可以设置

```c++
CxxConfig config;
config.set_model_dir("xxx");  // model_dir 为必须选项
// 设置有效的Place信息
config.set_valid_places({Place{TARGET(kARM), PRECISION(kFloat)}});
// 当每个Op有多个kernel可选择的时候，优先选择preferred_place可运行的kernel。
config.set_preferred_place(Place{TARGET(kARM), PRECISION(kInt8)});
```

 创建完 config 之后可以继续获得 predictor 来执行预测

```c++
auto predictor = CreatePaddlePredictor(config);
```

获取模型的输入和输出 tensor 以填充或获取数据。

这里的 Tensor 都是 handle，用户最好复用。

```c++
auto x_tensor = predictor->GetInput(0/*index*/);
// 这里的 0 表示输入序列的 offset，具体的顺序由训练中 save_inference_model 存储决定
// 注意，这里的 x_tensor 是一个 unique_ptr，也就是一个对应的 handle，用户可以在每个 batch 都复用
// 这个 handle.
auto out_tensor = predictor->GetOutput(0/*index*/);
// 这里 out_tensor 是只读的
```

 这里的 Tensor 提供了用户需要的详细的信息，其定义如下，用户可以自由使用其他接口

```c++
struct Tensor {
  void Resize(const shape_t& shape);

  /// Readonly data.
  template <typename T>
  const T* data() const;

  template <typename T>
  T* mutable_data() const;

  /// Shape of the tensor.
  shape_t shape() const;
};
```

接着上面例子，`x_tensor` 是第`0` 个输入的 Tensor，是可写的。 可以类似如下方式准备输入

```c++
// 指定 batch_size=10, 其余维度为 200, 30
// 注意，这里的维度需要参考实际模型做修改
x_tensor->Resize({10, 200, 30});
// Resize 更新 shape 后，调用 mutable_data 来实际分配内存
auto x_data = x_tensor->mutable_data<float>();
// 可以随意修改 x_data 的输入，比如 memcpy(x_data, some_data, some_size);
```

模型可能有多个输入，如上类似 `x_tensor` ，调用 `GetInput(i)` 获得其余 tensor 并修改。

输入准备完毕，就可以执行预测：

```c++
// 执行模型的预测，模型会基于前面设定的 input tensor，执行模型计算，并填充 output tensor
predictor->Run();
```

 执行完毕，可以获取 output tensor 的数据

```c++
// 获得 output tensor 的 shape
auto out_shape = out_tensor->shape();

// 获得具体的 data，是一块连续的 memory
const auto* out_data = out_tensor->data<float>();
```

### MobileConfig

`MobileConfig` 基本用法类似于 `CxxConfig` ，具体区别是

- CxxConfig 会执行完整的预测，包括图分析等较重的逻辑
  - 输入为原始的预测模型，无需做离线处理
  - 可以将图分析优化完的模型存储下来（借助 SaveOptimizedModel 接口），用于 `MobileConfig`
- MobileConfig 考虑到手机应用的空间及初始化时长的限制，阉割掉图分析的能力，只执行预测本身
  - 更轻量级
  - 输入模型必须为图分析优化完的模型 (借助 CxxConfig 作离线处理)

由于 MobileConfig 的输入模型必须为优化完的模型，相应的 Kernel 的 Place 由输入模型决定，因此没有 CxxConfig 中 指定Place的接口，目前只有指定模型路径的接口：

-  `void set_model_dir(const std::string& x)`

使用 MobileConfig 的其余步骤 与CxxConfig 完全一致。

### GenCode 功能介绍

Mobile 支持将模型和预测库结合，转化为 C++代码，进而融合成一个链接库，在设备上执行`paddle_code_generator` 及相应参数便可转化。

### INT8量化预测

Paddle-Mobile支持对[PaddleSlim](https://github.com/PaddlePaddle/models/tree/develop/PaddleSlim)中量化训练得到的模型的预测。

其中使用方法如下：

```c++
CxxConfig config;
config.set_model_dir("xxx");  // model_dir 为必须选项
// 由于 ARM Int8 模式只包括 Conv，MUL 等少数量化 kernel，因此需要一并选择上 Float 的 kernel
config.set_valid_places({Place{TARGET(kARM), PRECISION(kInt8)},  // Int8 计算 kernel
                         Place{TARGET(kARM), PRECISION(kFloat)}  // Float 也需要选择以补充
                        });
// 上面同时选择了 kInt8 和 kFloat 两类模式的 kernel，下面设置 kInt8 的 kernel 为优先选择
config.set_preferred_place(Place{TARGET(kARM), PRECISION(kInt8)});
```

目前该功能已在Mobilenetv1上进行了验证，并且还在持续开发中。


## 源码编译

### ARM CPU

当前ARM 上可以支持arm v8和v7的交叉编译。环境可以直接使用`paddle/fluid/lite/tools/Dockerfile.mobile`生成docker镜像。

- 主要的cmake选项
                
    - `ARM_MATH_LIB_DIR` 代表arm相关数学库的路径，可以从官网指定路径下载。
    - `ARM_TARGET_OS` 代表目标操作系统， 目前支持 "android" "armlinux"， 默认是Android
    - `ARM_TARGET_ARCH_ABI` 代表ARCH，支持输入"armv8"和"armv7"，针对OS不一样选择不一样。
        - `-DARM_TARGET_OS="android"` 时 
            - "armv8", 等效于 "arm64-v8a"。 default值为这个。
            - "armv7", 等效于 "armeabi-v7a"。 
        - `-DARM_TARGET_OS="armlinux"` 时 
            - "armv8", 等效于 "arm64"。 default值为这个。当前仅支持这个输入。
    - `ARM_TARGET_LANG` 代表目标编译的语言， 默认为gcc，支持 gcc和clang两种。

- 参考示例
	
	```shell
	# ARM_TARGET_OS in "android" , "armlinux"
	# ARM_TARGET_ARCH_ABI in "armv8", "armv7" ,"armv7hf"
	# ARM_TARGET_LANG in "gcc" "clang"
	cmake .. \
	    -DWITH_GPU=OFF \
	    -DWITH_MKL=OFF \
	    -DWITH_LITE=ON \
	    -DLITE_WITH_CUDA=OFF \
	    -DLITE_WITH_X86=OFF \
	    -DLITE_WITH_ARM=ON \
	    -DLITE_WITH_LIGHT_WEIGHT_FRAMEWORK=ON \
        -DARM_MATH_LIB_DIR="<to_arm_math_libs_path>" \
	    -DWITH_TESTING=ON \
	    -DARM_TARGET_OS="android" -DARM_TARGET_ARCH_ABI="armv8" -DARM_TARGET_LANG="gcc"
	make -j4
	```

### OpenCL

Paddle-Mobile支持在Android系统上运行基于OpenCL的程序，目前提供armv8和armv7的交叉编译。

#### 编译

- 编译环境: 使用`paddle/fluid/lite/tools/Dockerfile.mobile`生成docker镜像。
- cmake编译选型介绍
    * `ARM_TARGET_OS` 代表目标操作系统， 目前仅支持 "android", 亦为默认值。
    * `ARM_TARGET_ARCH_ABI` 代表ARCH，支持输入"armv8"和"armv7"。其中，"armv8",
    等效于 "arm64-v8a"，亦为默认值；"armv7", 等效于 "armeabi-v7a"。
    * `ARM_TARGET_LANG` 代表目标编译的语言， 默认为gcc，支持 gcc和clang两种。
- 参考示例

	```shell
	# ARM_TARGET_OS in "android"
	# ARM_TARGET_ARCH_ABI in "armv8", "armv7" ,"armv7hf"
	# ARM_TARGET_LANG in "gcc" "clang"
	# 假设我们处于源码根目录下
	mkdir build_opencl && cd build_opencl
	cmake .. \
	    -DLITE_WITH_OPENCL=ON \
	    -DWITH_GPU=OFF \
	    -DWITH_MKL=OFF \
	    -DWITH_LITE=ON \
	    -DLITE_WITH_CUDA=OFF \
	    -DLITE_WITH_X86=OFF \
	    -DLITE_WITH_ARM=ON \
	    -DLITE_WITH_LIGHT_WEIGHT_FRAMEWORK=ON \
	    -DWITH_TESTING=ON \
	    -DARM_TARGET_OS="android" -DARM_TARGET_ARCH_ABI="armv8" -DARM_TARGET_LANG="gcc"
	# 完整编译
	make -j4
	# 或者我们也可以make某一target文件
	make test_mobilenetv1_lite -j4
	make test_cl_runtime -j4
	make test_elementwise_add_opencl -j4
	make test_pool_opencl -j4
	```

#### 运行

- 运行文件准备

使用如下命令将运行OpenCL程序时需要加载的文件push到手机端(假设我们处于源码根目录下)：

```
# 我们将文件统一push到/data/local/tmp/opencl目录下
adb shell mkdir -p /data/local/tmp/opencl
# 将OpenCL的kernels文件push到/data/local/tmp/opencl目录下
adb push paddle/fluid/lite/opencl/cl_kernel /data/local/tmp/opencl
# 将mobilenet_v1的模型文件push到/data/local/tmp/opencl目录下
adb push build_opencl/third_party/install/mobilenet_v1 /data/local/tmp/opencl
# 将OpenCL测试程序(如test_mobilenetv1_lite) push到/data/local/tmp/opencl目录下
adb push paddle/fluid/lite/api/test_mobilenetv1_lite /data/local/tmp/opencl
```

- 运行OpenCL程序

使用如下命令运行OpenCL程序。其中，`--cl_path`指定了OpenCL的kernels文件即cl\_kernel所在目录，
`--modle_dir`指定了模型文件所在目录。

```shell
adb shell
cd /data/local/tmp/opencl && ./test_mobilenetv1_lite --cl_path=. --model_dir=mobilenet_v1
```

