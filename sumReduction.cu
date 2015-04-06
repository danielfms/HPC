#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE 4

__global__ void reduce(int *g_idata, int *g_odata) {
	__shared__ int sdata[BLOCK_SIZE];
    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    sdata[tid] = g_idata[i];
    __syncthreads();
    // do reduction in shared mem
    for(unsigned int s=blockDim.x/2; s > 0; s >>= 1) {
        if(tid < s){
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

void llenar2(int *A,int N,int a){
	for(int  i = a; i <N; i++ )
        A[i] = 0;
}

void imprimir(int *A,int N){
	for(int i = 0; i <N; i++)
        printf("%d ",A[i]);
	printf("\n");
}



int main(){
  int N=19; //2*10240;
  int s;
  int bytes=(N)*sizeof(int);
  int *A=(int*)malloc(bytes);
  int *R=(int*)malloc(bytes);

	
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
  int *d_A=(int*)malloc(bytes);
  int *d_R=(int*)malloc(bytes);
  //cudaMalloc(&d_A,bytes);
  //cudaMalloc(&d_R,bytes);
  cudaMalloc((void**)&d_A,bytes);
  cudaMalloc((void**)&d_R,bytes);
  cudaMemcpy(d_A, A, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_R, R, bytes, cudaMemcpyHostToDevice);
  float blocksize=BLOCK_SIZE;
  clock_t start2 = clock();  
  int i=N;
  while(i>1){
  
    dim3 dimBlock(BLOCK_SIZE,1,1);
    int grid=ceil(i/blocksize);
    dim3 dimGrid(grid,1,1);
    
    reduce<<<dimGrid,dimBlock>>>(d_A,d_R);
    cudaDeviceSynchronize();
    cudaMemcpy(R, d_R, bytes, cudaMemcpyDeviceToHost);
    //imprimir(R,N);
    llenar2(R,N,grid); //Lleno de ceros lo que no tendre en cuenta.
    cudaMemcpy(d_A,R, bytes, cudaMemcpyHostToDevice);
    i=ceil(i/blocksize);
    std::cout<<i<<std::endl;
  }
  cudaMemcpy(R, d_R, bytes, cudaMemcpyDeviceToHost);

  
  /////////////////////Llamado multikernel///////////////////////
 /*for(int i=1;i<N;i*=BLOCK_SIZE){
    reduce<<<dimGrid,dimBlock>>>(d_R,d_A);
  	cudaDeviceSynchronize();
  	cudaMemcpy(d_R, d_A, bytes, cudaMemcpyDeviceToHost);  
  } */
  ////////////////////Llamado multikernel///////////////////////


  
  // Copy array back to host
 
  clock_t end2= clock(); 
  double elapsed_seconds2=end2-start2;
  printf("Tiempo transcurrido Paralelo Reduccion: %lf\n", (elapsed_seconds2 / CLOCKS_PER_SEC));  
	
  /////////////////////////////////////////////////////////////
  if(s==R[0])
    printf("Las sumatorias son iguales: %d %d \n",s,R[0]);
  else
    printf("Las sumatorias no son iguales: %d %d \n",s,R[0]);
  
  free(A);
  free(R);
  cudaFree(d_A);
  cudaFree(d_R);
  return 0;
}
