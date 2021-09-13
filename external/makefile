nvcc = /global/software/sl-7.x86_64/modules/langs/cuda/10.0/bin/nvcc
cudalib =/global/software/sl-7.x86_64/modules/langs/cuda/10.0/lib64
tensorflow = /global/home/users/fanwei_kong/tf1/lib/python3.6/site-packages/tensorflow/include
TF_LIB= /global/home/users/fanwei_kong/tf1/lib/python3.6/site-packages/tensorflow
all: tf_nndistance_so.so tf_nndistance_g.cu.o 

tf_nndistance_so.so: tf_nndistance_g.cu.o tf_nndistance.cpp
	g++ -std=c++11 tf_nndistance.cpp tf_nndistance_g.cu.o -o tf_nndistance_so.so -shared -fPIC -I $(tensorflow) -lcudart -L $(cudalib) -O2 -L $(TF_LIB) -ltensorflow_framework -D_GLIBCXX_USE_CXX11_ABI=0

tf_nndistance_g.cu.o: tf_nndistance_g.cu
	$(nvcc) -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++11 -c -o tf_nndistance_g.cu.o tf_nndistance_g.cu -I $(tensorflow) -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -O2

clean:
	rm tf_nndistance_so.so
	rm  *.cu.o 
