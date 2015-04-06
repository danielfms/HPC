#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE 32

__global__ void reduce0(int *g_idata, int *g_odata) {
	extern __shared__ int sdata[];
	// each thread loads one element from global to shared mem
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
	sdata[tid] = g_idata[i];
	__syncthreads();
	// do reduction in shared mem
	for(unsigned int s=1; s < blockDim.x; s *= 2) {
		if (tid % (2*s) == 0) {
		sdata[tid] += sdata[tid + s];
		}
	__syncthreads();
	}
	// write result for this block to global mem
	if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}


void sumS(int *A,int N ,int *r){
	int value=0;
	for(int i=0;i<N;i++)
		value+=A[i];
	*r=value;
  
}

void llenar(int *A,int N,int a){
	for(int  i = 0; i <N; i++ )
        A[i] = a;
}

void imprimir(int *A,int N){
	for(int i = 0; i <N; i++)
        printf("%d ",A[i]);
  
	printf("\n");
}



int main(){
	int N=20;
	size_t bytes=(N)*sizeof(int);
	int *A=(int*)malloc(bytes);
    int *R=(int*)malloc(bytes);
	int s;
	
    //Lleno las matrices.
	llenar(A,N,1);
	llenar(R,N,0);
 
  
    //////////////////Algoritmo secuencial///////////////////////
	clock_t start = clock();      
    sumS(A,N,&s);
    clock_t end= clock(); 
	double elapsed_seconds=end-start;
    printf("Tiempo transcurrido Secuencial: %lf\n", (elapsed_seconds / CLOCKS_PER_SEC));
    /////////////////////////////////////////////////////////////
  
  
    ////////////////////////Algoritmo Paralelo///////////////////
  
    //////Separo memoria para el algoritmo paralelo
	int *d_A;
    int *d_R;
    cudaMalloc(&d_A,bytes);
    cudaMalloc(&d_R,bytes);

    cudaMemcpy(d_A, A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_R, R, bytes, cudaMemcpyHostToDevice);
  
    float blocksize=BLOCK_SIZE;
	dim3 dimGrid(ceil(N/blocksize),1,1);
    dim3 dimBlock(blocksize,1,1);
	
	clock_t start2 = clock();  
	reduce0<<<dimGrid,dimBlock>>>(d_A,d_R);
	cudaDeviceSynchronize();
    // Copy array back to host
    cudaMemcpy(R,d_R, bytes, cudaMemcpyDeviceToHost );
    clock_t end2= clock(); 
	double elapsed_seconds2=end2-start2;
    printf("Tiempo transcurrido Paralelo Reduccion: %lf\n", (elapsed_seconds2 / CLOCKS_PER_SEC));  
	
    /////////////////////////////////////////////////////////////
  
	if(s!=R[0])
		printf("Las sumatorias no son iguales: %d %d \n",s,R[0]);
	
    for(int i=0;i<N;i++)
   		printf("%d ",R[i]);
   
	return 0;
}
