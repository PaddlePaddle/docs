***
# **Compile From Source Code Under Windows**

This instruction will show you how to compile PaddlePaddle on a *64-bit desktop or laptop* and Windows 10. The Windows systems we support must meet the following requirements:

* Windows 10 Family Edition / Professional Edition / Enterprise Edition
* Visual Studio 2015 Update3

## Determine which version to compile

* **Only PaddlePaddle for CPU is supported.**

## Choose a compilation method

We provide one compilation method under the Windows system:

* Direct source code compilation

Since the situation on host machine is more complicated, we only support specific systems.

Please note: The current version does not support NCCL, distributed, AVX, warpctc and MKL related functions.


### ***Local compilation***

**Please strictly follow the following instructions step by step**

1. Check that your computer and operating system meet our supported compilation standards

	* Windows 10 Family Edition / Professional Edition / Enterprise Edition

	* Visual Studio 2015 Update3

2. Install the necessary tools i.e. cmake, git and python :

	> Cmake requires version 3.0 and above, which can be downloaded from the official website and added to the environment variable. [Download here](https://cmake.org/download/).

	> Git can be downloaded on the official website and added to the environment variable. [Download here](https://gitforwindows.org/).

	> Python requires version 2.7 and above, and ensure that modules such as numpy, protobuf, wheel are installed. [Download here](https://www.python.org/download/releases/2.7/).


		* To Install numpy package you can use command `pip install numpy` or command `pip3 install numpy`

		* To Install protobuf package you can use command `pip install protobuf` or command `pip3 install protobuf`

		* To Install Wheel package you can use command `pip install wheel` or `pip3 install wheel`


3. Clone the PaddlePaddle source in the Paddle folder in the current directory and go to the Paddle directory:

	- `git clone https://github.com/PaddlePaddle/Paddle.git`
	- `cd Paddle`

4. Switch to a more stable release branch for compilation (supports 1.2.x and above):

	- `git checkout release/x.x.x`

5. Create a directory called build and enter it:

	- `mkdir build`
	- `cd build`

6. Execute cmake:

	> For details on the compilation options, see [the compilation options list](../Tables.html/#Compile).

	* For users who need to compile **the CPU version PaddlePaddle**:

		For Python2:`cmake .. -G "Visual Studio 14 2015 Win64" -DPYTHON_INCLUDE_DIR = $ {PYTHON_INCLUDE_DIRS} 
			-DPYTHON_LIBRARY = $ {PYTHON_LIBRARY} 
			-DPYTHON_EXECUTABLE = $ {PYTHON_EXECUTABLE} -DWITH_FLUID_ONLY = ON -DWITH_GPU = OFF -DWITH_TESTING = OFF -DCMAKE_BUILD_TYPE =Release`


		For Python3: `cmake .. -G "Visual Studio 14 2015 Win64" -DPY_VERSION = 3.5 -DPYTHON_INCLUDE_DIR = $ {PYTHON_INCLUDE_DIRS} 
			-DPYTHON_LIBRARY = $ {PYTHON_LIBRARY} 
			-DPYTHON_EXECUTABLE = $ {PYTHON_EXECUTABLE} -DWITH_FLUID_ONLY = ON -DWITH_GPU = OFF -DWITH_TESTING =OFF -DCMAKE_BUILD_TYPE=Release`

		> If you encounter `Could NOT find PROTOBUF (missing: PROTOBUF_LIBRARY PROTOBUF_INCLUDE_DIR)`, you can re-execute the cmake command.

7. Some third-party dependencies (openblas, snappystream) currently require users to provide pre-compiled versions, or download pre-compiled files from `https://github.com/wopeizl/Paddle_deps` and place the entire `third_party` folder in the `build` directory. 

8. Use Blend for Visual Studio 2015 to open `paddle.sln` file, select the platform `x64`, configure with `Release`, then begin to compile

9. Having compiled successfully, go to the `\paddle\build\python\dist`directory and find the generated `.whl` package:

	`cd \paddle\build\python\dist`

10. Install the compiled `.whl` package on the current machine or target machine:

 	`pip install (whl package name)` or `pip3 install (whl package name)`

Congratulations, now you have completed the process of compiling PaddlePaddle natively.


### ***Verify installation***

After the installation is complete, you can use: `python` to enter the Python interpreter and then use `import paddle.fluid`. If there is no error prompted, the installation is successful.

### ***How to uninstall***

Please use the following command to uninstall PaddlePaddle:

* ***CPU version of PaddlePaddle*** : `pip uninstall paddlepaddle` or `pip3 uninstall paddlepaddle`
