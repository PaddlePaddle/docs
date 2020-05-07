#!/bin/bash

# fluid
for module in layers dataset clip metrics executor initializer io nets optimizer profiler regularizer transpiler backward profiler unique_name dygraph
do
  python gen_doc.py --module_name ${module} --module_prefix ${module} --output ${module} --output_name fluid --to_multiple_files True
  python gen_module_index.py ${module}  fluid.${module}
done

python gen_doc.py --module_name "" --module_prefix "" --output fluid --output_name fluid --to_multiple_files True
python gen_module_index.py fluid  fluid

# tensor
for module in math random stat
do
  python gen_doc.py --module_name ${module} --module_prefix ${module} --output ${module} --output_name tensor --to_multiple_files True --output_dir tensor
  python gen_module_index.py tensor.${module} ${module}
done

python gen_module_index.py tensor paddle.tensor

for module in math manipulation
do
  python gen_doc.py --module_name tensor.${module} --module_prefix tensor.${module} --output tensor/${module} --output_name complex --to_multiple_files True --output_dir complex
  python gen_module_index.py complex.tensor.${module} ${module}
done

python gen_module_index.py complex.tensor tensor
python gen_module_index.py complex paddle.complex

# nn
for module in loss
do
  python gen_doc.py --module_name ${module} --module_prefix ${module} --output ${module} --output_name nn --to_multiple_files True --output_dir nn
  python gen_module_index.py nn.${module} ${module}
done

python gen_doc.py --module_name "" --module_prefix "" --output nn --output_name nn --to_multiple_files True
python gen_module_index.py nn paddle.nn

# index.rst
python gen_index.py

