..  _install_introduction_:

=======================
 Installation Guide
=======================


----------------------
  Important updates
----------------------

* Paddle supports user installation without depending on CUDA and cuDNN, and automatically handles the installation of CUDA and cuDNN.


------------------------
  Installation Manuals
------------------------


The manuals will guide you to build and install PaddlePaddle on your 64-bit desktop or laptop.

1. Operating system requirements:
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

* Windows 7 / 8 / 10 / 11, Pro/Enterprise
* Ubuntu 20.04 / 22.04
* almalinux 8
* macOS 12.x/13.x/14.x/15.x
* 64-bit operating system is required

2. Processor requirements:
>>>>>>>>>>>>>>>>>>>>>>>>>>

* Processor supports MKL
* The processor architecture is x86_64(or called x64, Intel 64, AMD64). Currently, PaddlePaddle does not support arm64.

3. Version requirements of python and pip:
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

* Python requires version 3.9/3.10/3.11/3.12/3.13
* Python needs pip, and pip requires version 20.2.2 or above
* Python and pip requires 64-bit

**First Installation Method: Using pip for installation**
You can choose any of the four methods: "Using pip for installation", "Using docker for installation", "Compiling from source code for installation".

This section will introduce the installation method using pip.

1.You need to ensure that your operating system meets the requirements listed above.

2.You need to ensure that your processor meets the requirements listed above.

3.Ensure that the Python where you need to install PaddlePaddle is in your expected location, as your computer may have multiple Pythons.

    Use the following command to output the Python path, depending on your environment you may need to replace all the python in the command line in the instructions with the specific Python path.

      In the Windows environment, the command to output the Python path is:

      ::

          where python

      In the macOS/Linux environment, the command to output the Python path is:

      ::

          which python

4.Check the Python version

    Use the following command to confirm it is 3.9/3.10/3.11/3.12/3.13

    ::

        python --version

5.Check the pip version, confirm it is 20.2.2+

    ::

        python -m ensurepip
        python -m pip --version

6.Confirm that Python and pip are 64 bit, and the processor architecture is x86_64 (also known as x64, Intel 64, AMD64). The first line of the following output is "64bit", and the second line output is "x86_64", "x64" or "AMD64":

    ::

        python -c "import platform;print(platform.architecture()[0]);print(platform.machine())"

7.If you want to use pip <https://pypi.org/project/pip/>_ to install PaddlePaddle, you can directly use the following command:

    Note:

      * If you want to install paddlepaddle, the version requires libstdc++.so.6 version greater than 3.4.25. To meet this requirement, you can choose to install GCC 8 or a higher GCC version, or upgrade the libstdc++ library separately. *

    (1). **CPU version** : If you just want to install the CPU version, please refer to the following command for installation

         The command to install the CPU version is:
         ::
               python -m pip install --pre paddlepaddle -i https://www.paddlepaddle.org.cn/packages/nightly/cpu/

    (2). **GPU version** : If you want to use the GPU version, please refer to the following command for installation

         The command to install the GPU cuda12.9 version is:
         ::
               python -m pip install --pre paddlepaddle-gpu -i https://www.paddlepaddle.org.cn/packages/nightly/cu129/

         The command to install the GPU cuda12.6 version is:
         ::
               python -m pip install --pre paddlepaddle-gpu -i https://www.paddlepaddle.org.cn/packages/nightly/cu126/

         The command to install the GPU cuda11.8 version is:
         ::
               python -m pip install --pre paddlepaddle-gpu -i https://www.paddlepaddle.org.cn/packages/nightly/cu118/

      Please make sure that the Python where you need to install PaddlePaddle is in the expected location, as your computer may have multiple Pythons. Depending on your environment, you may need to replace all python in the command line in the instructions with the specific Python path.

8.Verify the installation

    Use python to enter the python interpreter, enter import paddle, then enter paddle.utils.run_check().

    If PaddlePaddle is installed successfully! appears, it means you have successfully installed.

9.For more help information, please refer to:

    `PIP Installation under Linux <pip/linux-pip.html>`_

    `PIP Installation under macOS <pip/macos-pip.html>`_

    `PIP Installation under Windows <pip/windows-pip.html>`_

  **Second Installation Method: Using Source Code Compilation for Installation**

  - If you are just using PaddlePaddle, it is recommended to use **pip** for installation.

  - If you have a need to develop PaddlePaddle, please refer to: `Compiling from Source <compile/fromsource.html>`_

..  toctree::
    :hidden:

    pip/frompip_en.rst
    docker/fromdocker_en.rst
    compile/fromsource_en.rst
    install_NGC_PaddlePaddle_en.rst
    Tables_en.md
