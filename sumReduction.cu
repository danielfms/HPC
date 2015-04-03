#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE 512 
#define NUM_OF_ELEMS 4096 

__global__ void sumP(int *A,int *r){
	__shared__ int partialSum[NUM_OF_ELEMS];
	int t = threadIdx.x;
	//int i = blockIdx.x*blockDim.x + threadIdx.x;
  
	//if(t<NUM_OF_ELEMS)
	partialSum[t] = A[t];//i
  __syncthreads();
	
	for (unsigned int stride = blockDim.x; stride > 1; stride /= 2){
		if (t < stride)
			partialSum[t] += partialSum[t+stride];
		__syncthreads();
	}
   __syncthreads();
  if (t == 0)
		r[0]=partialSum[t];
}


void sumS(int *A, int *r){
  int value=0;
	for(int i=0;i<NUM_OF_ELEMS;i++)
			value+=A[i];
  *r=value;
  
}

void llenar(int *A){
	for(int  i = 0; i <NUM_OF_ELEMS; i++ )
        A[i] = 1;
}

void imprimir(int *A){
	for(int i = 0; i <NUM_OF_ELEMS; i++)
        printf("%d ",A[i]);
  
	printf("\n");
}



int main(){
  int n=16;
	size_t bytes=(NUM_OF_ELEMS)*sizeof(int);
	int *A=(int*)malloc(bytes);
  int *R=(int*)malloc(bytes);
	int s;
	
	llenar(A);
  //imprimir(A);
	clock_t start = clock();      
  sumS(A,&s);
  clock_t end= clock(); 
	double elapsed_seconds=end-start;
  printf("Tiempo transcurrido Secuencial: %lf\n", (elapsed_seconds / CLOCKS_PER_SEC));
  
	int *d_A;
  int *d_R;
  cudaMalloc(&d_A,bytes);
  cudaMalloc(&d_R,bytes);

	cudaMemcpy(d_A, A, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_R, R, bytes, cudaMemcpyHostToDevice);

	//Number of threads in each thread block
		//bloques
  float blocksize=BLOCK_SIZE;
	dim3 dimGrid(ceil(NUM_OF_ELEMS/blocksize),1,1);
  //hilos
  dim3 dimBlock(blocksize,1,1);
	
	clock_t start2 = clock();  
	sumP<<<dimGrid,dimBlock>>>(d_A,d_R);

  // Copy array back to host
  cudaMemcpy(R,d_R, bytes, cudaMemcpyDeviceToHost );
  clock_t end2= clock(); 
	double elapsed_seconds2=end2-start2;
  printf("Tiempo transcurrido Paralelo Reduccion: %lf\n", (elapsed_seconds2 / CLOCKS_PER_SEC));  
	
	if(s!=R[0])
		printf("Las sumatorias no son iguales: %d %d \n",s,R[0]);
	
  for(int i=0;i<NUM_OF_ELEMS;i++)
   printf("%d ",R[i]);
   
	return 0;
}
