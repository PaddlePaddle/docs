.. _cn_api_fluid_io_PipeReader:

PipeReader
-------------------------------

.. py:class:: paddle.fluid.io.PipeReader(command, bufsize=8192, file_type="plain")


PipeReader从命令的输出流中读取数据，把数据存在一个pipe缓存中，并重定向到解析器中解析，返回预先设计格式的数据。 

读取的数据可以来自标准的Linux命令或者调用其他程序，从HDFS, Ceph, URL, AWS S3等等，如下是一些命令实例：

..  code-block:: python

           cmd = "hadoop fs -cat /path/to/some/file"
           cmd = "cat sample_file.tar.gz"
           cmd = "curl http://someurl"
           cmd = "python print_s3_bucket.py"
参数:    
    - **command** (str) – 该参数表示产生数据来源的命令。
    - **bufsize** (int) – 该参数表示pipe缓存的大小，默认为8192。
    - **file_type** (str) – command操作文件类型，默认为plain。

**代码示例**

..  code-block:: python

           import paddle
           def example_reader(filelist):
               for f in filelist:
                   pr = paddle.reader.PipeReader("cat %s"%f)
                   for l in pr.get_line():
                       sample = l.split(" ")
                       yield sample

.. py:method:: get_line(cut_lines=True,line_break='\n')

参数：
    - **cut_lines** (bool) - 给行分割缓存
    - **line_break** (string) - 行分隔符，比如'\n'或者'\r' 

返回： 行或者字节缓存
