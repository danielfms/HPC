#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE 32
using namespace std;

__global__ void KernelConvolutionBasic(int *N,int *M,int *P,int Mask_Width,int Width){
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

void compare(int*A,int *B,int width){
  for(int i=0;i<width;i++)
    if(A[i]!=B[i]){
    	cout<<"Los vectores no son iguales"<<endl;
    	return;
  	}
  cout<<"Los vectores son iguales"<<endl;	 
}

int main(){
  
  int N=7;
  int bytes=(N)*sizeof(int);
  int *V=(int*)malloc(bytes);
  int *P=(int*)malloc(bytes);
  int Mask[5]={3,4,5,4,3};
  
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
  
  //Variables para el kernel
  int *d_V;
  int *d_P;
  int *d_Mask;
  int bytesM=5*sizeof(int);
  int *P_out=(int*)malloc(bytes);
  int *P_in=(int*)malloc(bytes);
  llenar(P_in,N,0);
  
  cudaMalloc(&d_V,bytes);
  cudaMalloc(&d_P,bytes);
  cudaMalloc(&d_Mask,bytesM);

  cudaMemcpy(d_V, V, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_P, P_in, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_Mask, Mask, bytesM, cudaMemcpyHostToDevice);
  
  //Bloque y grid
  float blocksize=BLOCK_SIZE;
  dim3 dimGrid(ceil(N/blocksize),1,1);
  dim3 dimBlock(blocksize,1,1);
  
  //Convolucion Paralelo
  start=clock();
  KernelConvolutionBasic<<<dimGrid,dimBlock>>>(d_V,d_Mask,d_P,5,N);
  cudaDeviceSynchronize();
  cudaMemcpy(P_out,d_P, bytes, cudaMemcpyDeviceToHost );
  end=clock();
  double elapsed_seconds2=end-start;
  printf("Tiempo transcurrido Secuencial: %lf\n", (elapsed_seconds2 / CLOCKS_PER_SEC));
  //imprimirVec(P_out,N);
  //////////////////////
  
  compare(P,P_out,N);
  cout<<"Aceleracion obtenida: "<<elapsed_seconds/elapsed_seconds2<<endl;
  
  free(V);
  free(P);
  //free(Mask);
  free(P_in);
  free(P_out);
  cudaFree(d_V);
  cudaFree(d_P);
  
  return 0; 
}
