#!/bin/bash

for module in layers dataset clip metrics executor initializer io nets optimizer profiler regularizer transpiler backward profiler unique_name dygraph framework
do
  python gen_doc.py --module_name ${module} --module_prefix ${module} --output ${module} --output_name fluid --to_multiple_files True
  python gen_module_index.py ${module}  fluid.${module}
done

python gen_doc.py --module_name "" --module_prefix "" --output fluid --output_name fluid --to_multiple_files True
python gen_module_index.py fluid  fluid

# tensor
for module in math random stat linalg search
do
  python gen_doc.py --module_name ${module} --module_prefix ${module} --output ${module} --output_name tensor --to_multiple_files True --output_dir tensor
  python gen_module_index.py tensor.${module} ${module}
done

python gen_module_index.py tensor paddle.tensor

for module in math manipulation linalg
do
  python gen_doc.py --module_name tensor.${module} --module_prefix tensor.${module} --output tensor/${module} --output_name complex --to_multiple_files True --output_dir complex
  python gen_module_index.py complex.tensor.${module} ${module}
done

python gen_module_index.py complex.tensor tensor
python gen_module_index.py complex paddle.complex
python gen_module_index.py framework paddle.framework


# nn
for module in loss
do
  python gen_doc.py --module_name ${module} --module_prefix ${module} --output ${module} --output_name nn --to_multiple_files True --output_dir nn
  python gen_module_index.py nn.${module} ${module}
done

python gen_doc.py --module_name "" --module_prefix "" --output nn --output_name nn --to_multiple_files True
python gen_module_index.py nn paddle.nn

# hapi
for module in transforms models
do
  python gen_doc.py --module_name ${module} --module_prefix ${module} --output ${module} --output_name  incubate.hapi.vision --to_multiple_files True --output_dir  incubate/hapi/vision
  python gen_module_index.py incubate.hapi.vision.${module} ${module}
done

for module in text callbacks distributed download loss metrics model 
do
  python gen_doc.py --module_name ${module} --module_prefix ${module} --output ${module} --output_name  incubate.hapi --to_multiple_files True --output_dir  incubate/hapi
  python gen_module_index.py ${module} ${module}
done

python gen_module_index.py incubate.hapi.vision vision
python gen_module_index.py incubate.hapi hapi
python gen_module_index.py incubate paddle.incubate

# index.rst
python gen_index.py

