#NOTE: Modify the following variables according to the cuda version
#      and the current installation of python. the following bash
#      commands can help to find the right folders for the cuda/python
#      libaries.
#      TF_LIB=$(python3 -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')
#      tensorflow=$(python3 -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
#
#      useful link: https://github.com/nywang16/Pixel2Mesh/issues/32

nvcc = /usr/local/cuda-10.1/bin/nvcc
cudalib = /usr/local/cuda-10.1/lib64
cudalib_include = /usr/local/cuda-10.1/include
tensorflow=/home/diegopc/.local/lib/python3.6/site-packages/tensorflow/include
external_nsync = /home/diegopc/.local/lib/python3.6/site-packages/tensorflow/include/external/nsync/public
TF_LIB=/home/diegopc/.local/lib/python3.6/site-packages/tensorflow

all: tf_approxmatch_so.so tf_approxmatch_g.cu.o tf_nndistance_so.so tf_nndistance_g.cu.o tf_auctionmatch_so.so tf_auctionmatch_g.cu tf_sampling_so.so tf_sampling_g.cu.o

tf_approxmatch_so.so: tf_approxmatch_g.cu.o tf_approxmatch.cpp
	g++ -std=c++11 tf_approxmatch.cpp tf_approxmatch_g.cu.o -o tf_approxmatch_so.so -shared -fPIC -I $(tensorflow) -lcudart -L $(cudalib) -O2 -D_GLIBCXX_USE_CXX11_ABI=0


tf_approxmatch_g.cu.o: tf_approxmatch_g.cu
	$(nvcc) -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++11 -c -o tf_approxmatch_g.cu.o tf_approxmatch_g.cu -I $(tensorflow) -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -O2


tf_nndistance_so.so: tf_nndistance_g.cu.o tf_nndistance.cpp
	g++ -std=c++11 tf_nndistance.cpp tf_nndistance_g.cu.o -o tf_nndistance_so.so -shared -fPIC -I $(tensorflow) -lcudart -L $(cudalib) -O2 -L $(TF_LIB) -ltensorflow_framework -D_GLIBCXX_USE_CXX11_ABI=0


tf_nndistance_g.cu.o: tf_nndistance_g.cu
	$(nvcc) -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++11 -c -o tf_nndistance_g.cu.o tf_nndistance_g.cu -I $(tensorflow) -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -O2

tf_auctionmatch_so.so: tf_auctionmatch_g.cu.o tf_auctionmatch.cpp
	g++ -std=c++11 tf_auctionmatch.cpp tf_auctionmatch_g.cu.o -o tf_auctionmatch_so.so -shared -fPIC -I $(tensorflow) -I $(cudalib_include) -I $(external_nsync) -lcudart -L $(cudalib) -L $(TF_LIB) -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=0

tf_auctionmatch_g.cu.o: tf_auctionmatch_g.cu
	$(nvcc) tf_auctionmatch_g.cu -o tf_auctionmatch_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
	
tf_sampling_so.so: tf_sampling_g.cu.o tf_sampling.cpp
	g++ -std=c++11 tf_sampling.cpp tf_sampling_g.cu.o -o tf_sampling_so.so -shared -fPIC -I $(tensorflow) -I $(cudalib_include) -I $(external_nsync) -lcudart -L $(cudalib) -L $(TF_LIB) -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=0

tf_sampling_g.cu.o: tf_sampling_g.cu
	$(nvcc) tf_sampling_g.cu -o tf_sampling_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

clean:
	rm tf_approxmatch_so.so
	rm tf_nndistance_so.so
	rm tf_auctionmatch_so.so
	rm tf_sampling_so.so
	rm  *.cu.o 
