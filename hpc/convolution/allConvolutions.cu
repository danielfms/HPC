#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE 1024
#define MAX_MASK_WIDTH 5
#define TILE_SIZE 1024
__constant__ int M[MAX_MASK_WIDTH];

using namespace std;

__global__ void KernelConvolutionTile(int *N, int *P, int Mask_Width,int Width) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  __shared__ int N_ds[TILE_SIZE + MAX_MASK_WIDTH -1];
  int n = Mask_Width/2;
  
  int halo_index_left = (blockIdx.x - 1)*blockDim.x + threadIdx.x;
  if (threadIdx.x >= blockDim.x - n) {
    N_ds[threadIdx.x - (blockDim.x - n)] =(halo_index_left < 0) ? 0 : N[halo_index_left];
  }
  if(i<Width)
  	N_ds[n + threadIdx.x] = N[i];
  else
    N_ds[n + threadIdx.x] = 0;
  int halo_index_right = (blockIdx.x + 1)*blockDim.x + threadIdx.x;
  if (threadIdx.x < n) {
    N_ds[n + blockDim.x + threadIdx.x] =(halo_index_right >= Width) ? 0 : N[halo_index_right];
  }
  __syncthreads();
  int Pvalue = 0;
  for(int j = 0; j < Mask_Width; j++) {
    Pvalue += N_ds[threadIdx.x + j]*M[j];
  }
  P[i] = Pvalue;
}

__global__ void KernelConvolutionBasic(int *N,int *M,int *P,int Mask_Width,int Width){
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  int N_start_point = i - (Mask_Width/2);
    int Pvalue=0;
    for (int j= 0;j<Mask_Width;j++) {
      if (N_start_point+j >= 0 && N_start_point + j < Width) {
      Pvalue+=N[N_start_point+j]*M[j];
      }
    }
  	if(i<Width)
    	P[i]=Pvalue;
}

__global__ void KernelConvolutionCaching(int *N,int *P,int Mask_Width,int Width){
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  int N_start_point = i - (Mask_Width/2);
    int Pvalue=0;
    for (int j= 0;j<Mask_Width;j++) {
      if (N_start_point+j >= 0 && N_start_point + j < Width) {
      Pvalue+=N[N_start_point+j]*M[j];
      }
    }
    P[i]=Pvalue;
}


void convolutionBasic(int *N,int *M,int *P,int Mask_Width,int Width){
  for(int i=0;i<Width;i++){
    int N_start_point = i - (Mask_Width/2);
    int Pvalue=0;
    for (int j= 0;j<Mask_Width;j++) {
      if (N_start_point+j >= 0 && N_start_point + j < Width) {
      Pvalue+=N[N_start_point+j]*M[j];
      }
    }
    P[i]=Pvalue;
  }
}

void imprimirVec(int *V,int n){
  cout<<"|";
  for(int i=0;i<n;i++)
    cout<<V[i]<<"|";
  cout<<endl;
}

void llenar(int *V,int N,int flag){
  if(flag==1)
    for(int  i = 1; i <=N; i++ )
          V[i-1] = i;
  else
    for(int  i = 1; i <=N; i++ )
          V[i-1] = 0;
}

void compare(int*A,int *B1,int *B2,int *B3,int width){
  for(int i=0;i<width;i++)
    if(((A[i]!=B1[i])||(A[i]!=B2[i]))||(A[i]!=B3[i])){
      cout<<"Los vectores no son iguales"<<endl;
      return;
    }
  cout<<"Los vectores son iguales"<<endl;  
}

int main(){
  
  int N=62500000;
  int bytes=(N)*sizeof(int);
  int bytesM=MAX_MASK_WIDTH *sizeof(int);
  int *V=(int*)malloc(bytes);
  int *P=(int*)malloc(bytes);
  int Mask[MAX_MASK_WIDTH]={3,4,5,4,3};
  
  llenar(V,N,1);
  llenar(P,N,0);
  
  
  //Convolucion secuencial
  clock_t start = clock();      
  convolutionBasic(V,Mask,P,5,N);
  clock_t end= clock(); 
  double elapsed_seconds=end-start;
  printf("Tiempo transcurrido Secuencial: %lf\n", (elapsed_seconds / CLOCKS_PER_SEC));
  //imprimirVec(P,N);
  /////////////////////////
  
  
  
  //Bloque y grid
  float blocksize=BLOCK_SIZE;
  dim3 dimGrid(ceil(N/blocksize),1,1);
  dim3 dimBlock(blocksize,1,1);

  //////////////////////Convolucion Paralelo Basic////////////////////////
  //Variables para el kernel
  int *d_V1;
  int *d_P1;
  int *d_Mask1;
  int *P_out1=(int*)malloc(bytes);
  int *P_in1=(int*)malloc(bytes);
  llenar(P_in1,N,0);
  
 
  cudaMalloc(&d_V1,bytes);
  cudaMalloc(&d_P1,bytes);
  cudaMalloc(&d_Mask1,bytesM);

  start=clock();
  cudaMemcpy(d_V1, V, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_P1, P_in1, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_Mask1, Mask, bytesM, cudaMemcpyHostToDevice);
  
  
  KernelConvolutionBasic<<<dimGrid,dimBlock>>>(d_V1,d_Mask1,d_P1,MAX_MASK_WIDTH,N);
  cudaDeviceSynchronize();
  cudaMemcpy(P_out1,d_P1, bytes, cudaMemcpyDeviceToHost );
  end=clock();
  double elapsed_seconds1=end-start;
  printf("Tiempo transcurrido Paralelo Basic: %lf\n", (elapsed_seconds1 / CLOCKS_PER_SEC));
  //imprimirVec(P_out1,N);
  cout<<"Aceleracion obtenida: "<<elapsed_seconds/elapsed_seconds1<<endl<<endl;

  free(P_in1);
  cudaFree(d_V1);
  cudaFree(d_P1);

  //////////////////////////////////////////////////////////////////////////

  /////////////////////Convolucion Paralelo Caching/////////////////////////
  int *d_V2;
  int *d_P2;
  int *P_out2=(int*)malloc(bytes);
  int *P_in2=(int*)malloc(bytes);

  //Constant Memory
  cudaMemcpyToSymbol(M,Mask,bytesM);

  llenar(P_in2,N,0);
  
  
  cudaMalloc(&d_V2,bytes);
  cudaMalloc(&d_P2,bytes);

  start=clock();
  cudaMemcpy(d_V2, V, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_P2, P_in2, bytes, cudaMemcpyHostToDevice);

  
  KernelConvolutionCaching<<<dimGrid,dimBlock>>>(d_V2,d_P2,MAX_MASK_WIDTH,N);
  cudaDeviceSynchronize();
  cudaMemcpy(P_out2,d_P2, bytes, cudaMemcpyDeviceToHost );
  end=clock();
  double elapsed_seconds2=end-start;
  printf("Tiempo transcurrido Paralelo Caching: %lf\n", (elapsed_seconds2 / CLOCKS_PER_SEC));
  //imprimirVec(P_out2,N);
  cout<<"Aceleracion obtenida: "<<elapsed_seconds/elapsed_seconds2<<endl<<endl;
  free(P_in2);
  cudaFree(d_V2);
  cudaFree(d_P2);

  /////////////////////////////////////////////////////////////////////////////
  
  //////////////////////////Convolucion Paralelo Tile///////////////////////////

  //Variables para el kernel
  int *d_V3;
  int *d_P3;
  int *P_out3=(int*)malloc(bytes);
  int *P_in3=(int*)malloc(bytes);

  llenar(P_in3,N,0);
  
  
  cudaMalloc(&d_V3,bytes);
  cudaMalloc(&d_P3,bytes);

  start=clock();
  cudaMemcpy(d_V3, V, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_P3, P_in3, bytes, cudaMemcpyHostToDevice);

 
  KernelConvolutionTile<<<dimGrid,dimBlock>>>(d_V3,d_P3,MAX_MASK_WIDTH,N);
  cudaDeviceSynchronize();
  cudaMemcpy(P_out3,d_P3, bytes, cudaMemcpyDeviceToHost );
  end=clock();
  double elapsed_seconds3=end-start;
  printf("Tiempo transcurrido Paralelo Tile: %lf\n", (elapsed_seconds3 / CLOCKS_PER_SEC));
  //imprimirVec(P_out3,N);
  cout<<"Aceleracion obtenida: "<<elapsed_seconds/elapsed_seconds3<<endl;

  free(P_in3);
  cudaFree(d_V3);
  cudaFree(d_P3);

  /////////////////////////////////////////////////////////////////////////////
  
  compare(P,P_out1,P_out2,P_out3,N);
  free(V);
  free(P);
  free(P_out1);
  free(P_out2);
  free(P_out3);
  //free(Mask);

  return 0; 
}
