.. _cn_api_fluid_io_PipeReader:

PipeReader
-------------------------------

.. py:class:: paddle.fluid.io.PipeReader


PipeReader从命令中通过流来读取数据，把数据存在一个pipe缓存中，并重定向到解析器中解析，返回预先设计格式的数据。 

你可以使用标准的Linux命令或者调用其他程序来读取数据，从HDFS, Ceph, URL, AWS S3等等。

.. code-block:: python
           cmd = "hadoop fs -cat /path/to/some/file"
           cmd = "cat sample_file.tar.gz"
           cmd = "curl http://someurl"
           cmd = "python print_s3_bucket.py"


**代码示例**

.. code-block:: python
           def example_reader():
               for f in myfiles:
                   pr = PipeReader("cat %s"%f)
                   for l in pr.get_line():
                       sample = l.split(" ")
                       yield sample

.. py:method:: get_line(cut_lines=True,line_break='\n')

参数：
    - **cut_lines** (bool) - 给行分割缓存
    - **line_break** (string) - 行分隔符，比如'\n'或者'\r' 

返回： 行或者字节缓存
