ZMQ=/home/cardiac/zmq
ZMQ_LIBS=$(ZMQ)/lib
ZMQ_HDRS=$(ZMQ)/include

CC=/usr/local/cuda/bin/nvcc
CFLAGS= -I/usr/lib64/openmpi/1.2.7-gcc/include -I/usr/local/cuda/include -Xcompiler -fopenmp
LDFLAGS= -L/usr/lib64/openmpi/1.2.7-gcc/lib -L/usr/local/cuda/lib
LIB= -lgomp -lzmq -lczmq -lpthread -std=c++11
SOURCES= client.cu
SOURCES2= nodo.cu
EXECNAME= client
EXECNAME2= nodo
all:
	$(CC) -o $(EXECNAME) $(SOURCES) $(LIB) $(LDFLAGS) $(CFLAGS) -I$(ZMQ_HDRS) -L$(ZMQ_LIBS)
	$(CC) -o $(EXECNAME2) $(SOURCES2) $(LIB) $(LDFLAGS) $(CFLAGS) -I$(ZMQ_HDRS) -L$(ZMQ_LIBS)

client:
	$(CC) -o $(EXECNAME) $(SOURCES) $(LIB) $(LDFLAGS) $(CFLAGS) -I$(ZMQ_HDRS) -L$(ZMQ_LIBS)

node:
	$(CC) -o $(EXECNAME2) $(SOURCES2) $(LIB) $(LDFLAGS) $(CFLAGS) -I$(ZMQ_HDRS) -L$(ZMQ_LIBS)

clean:
	rm *.o *.linkinfo
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/cardiac/zmq/lib
