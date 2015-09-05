CUDNN_L=/home/nlg-05/dy_052/.local/lib  # where to find libcudnn.so
CUDNN_I=/home/nlg-05/dy_052/.local/lib  # where to find cudnn.h
CFLAGS=-O3 -L ${CUDNN_L} -I ${CUDNN_I}

libkudnn.so: kudnn.o  pool.o  xcorr.o
	nvcc $(CFLAGS) --shared --compiler-options -fPIC $^ -lcudnn -o $@

%.o: %.cu
	nvcc -c $(CFLAGS) --compiler-options -fPIC $< -o $@

clean:
	rm -f *.so
	rm -f *.o
