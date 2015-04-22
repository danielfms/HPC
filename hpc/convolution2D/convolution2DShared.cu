#include <iostream>
#include <highgui.h>
#include <cv.h>
#include <stdio.h>
#include <stdlib.h>

#define TILE_SIZE 32
#define MASK_WIDTH 9
#define Mask_size 3
__constant__ char M[MASK_WIDTH];



using namespace std;
using namespace cv;

__device__ unsigned char conv(int v){
  if(v>255)
    return 255;
  else if(v<0)
    return 0;
    
  return v;
}

__global__ void KernelConvolutionBasic(unsigned char *In, unsigned char *Out,int maskWidth, int width, int height){
  __shared__ float N_ds[TILE_SIZE + Mask_size - 1][TILE_SIZE+ Mask_size - 1];
   int n = Mask_size/2;
   int dest = threadIdx.y*TILE_SIZE+threadIdx.x, destY = dest / (TILE_SIZE+Mask_size-1), destX = dest % (TILE_SIZE+Mask_size-1),
       srcY = blockIdx.y * TILE_SIZE + destY - n, srcX = blockIdx.x * TILE_SIZE + destX - n,
       src = (srcY * width + srcX);
   if (srcY >= 0 && srcY < height && srcX >= 0 && srcX < width)
       N_ds[destY][destX] = In[src];
   else
       N_ds[destY][destX] = 0;

   // Second batch loading
   dest = threadIdx.y * TILE_SIZE + threadIdx.x + TILE_SIZE * TILE_SIZE;
   destY = dest /(TILE_SIZE + Mask_size - 1), destX = dest % (TILE_SIZE + Mask_size - 1);
   srcY = blockIdx.y * TILE_SIZE + destY - n;
   srcX = blockIdx.x * TILE_SIZE + destX - n;
   src = (srcY * width + srcX);
   if (destY < TILE_SIZE + Mask_size - 1) {
       if (srcY >= 0 && srcY < height && srcX >= 0 && srcX < width)
           N_ds[destY][destX] = In[src];
       else
           N_ds[destY][destX] = 0;
   }
   __syncthreads();

   int accum = 0;
   int y, x;
   for (y = 0; y < maskWidth; y++)
       for (x = 0; x < maskWidth; x++)
           accum += N_ds[threadIdx.y + y][threadIdx.x + x] * M[y * maskWidth + x];
   y = blockIdx.y * TILE_SIZE + threadIdx.y;
   x = blockIdx.x * TILE_SIZE + threadIdx.x;
   if (y < height && x < width)
       Out[(y * width + x)] = conv(accum);
   __syncthreads();
}


int main(){


  int scale = 1;
  int delta = 0;
  int ddepth = CV_8UC1;

  clock_t start,end; 
  double elapsed_seconds;  

  Mat image;
  //Leer imagen en escala de grises
  image = imread("inputs/img1.jpg",0);
  Size s = image.size();
  int row=s.width;
  int col=s.height;
  char Mask[9] = {-1,0,1,-2,0,2,-1,0,1};
  //imwrite("./outputs/1089746672.png",image);
  
  //Separo memoria para las imagenes en el host
  int sizeM= sizeof(char)*9;
  int size = sizeof(unsigned char)*row*col;
  unsigned char *img=(unsigned char*)malloc(size);
  unsigned char *img_out=(unsigned char*)malloc(size);

  img=image.data;

  /////////////////////////SECUENCIAL///////////////////////////////////////////

  Mat grad_x, grad_y;
  start=clock();
  Sobel( image, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
  end= clock(); 

  elapsed_seconds=end-start;
  printf("Tiempo transcurrido Secuencial: %lf\n", (elapsed_seconds / CLOCKS_PER_SEC));

  //////////////////////////////////////////////////////////////////////////////

  /////////////////////////////////PARALELO/////////////////////////////////////
 
  //Grid y blocksize
  float blocksize=32;
  dim3 dimBlock((int)blocksize,(int)blocksize,1);
  dim3 dimGrid(ceil(row/blocksize),ceil(col/blocksize),1);

   //Separo memoria en el device
  unsigned char *d_img;
  unsigned char *d_img_out;
  cudaMalloc((void**)&d_img,size);
  cudaMalloc((void**)&d_img_out,size);

  start=clock();
  cudaMemcpyToSymbol(M,Mask,sizeM);
  cudaMemcpy(d_img,img,size, cudaMemcpyHostToDevice);

  // Llamado al kernel
  KernelConvolutionBasic<<<dimGrid,dimBlock>>>(d_img,d_img_out,3,row,col);
  cudaDeviceSynchronize();
  cudaMemcpy(img_out,d_img_out,size,cudaMemcpyDeviceToHost);
  end=clock();

  elapsed_seconds=end-start;
  printf("Tiempo transcurrido Parelo: %lf\n", (elapsed_seconds / CLOCKS_PER_SEC));

  //Creo la imagen
  Mat gray_image;
  gray_image.create(col,row,CV_8UC1);
  gray_image.data = img_out;
  imwrite("./outputs/1089746672.png",gray_image);
  /////////////////////////////////////////////////////////////////////////////


  cudaFree(d_img);
  cudaFree(d_img_out);

  return 0; 
}
