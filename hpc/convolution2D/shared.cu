#include <iostream>
#include <highgui.h>
#include <cv.h>
#include <stdio.h>
#include <stdlib.h>


#define TILE_WIDTH 32



using namespace std;
using namespace cv;

__device__ unsigned char conv(int v){
  if(v>255)
    return 255;
  else if(v<0)
    return 0;
    
  return v;
}

__global__ void KernelConvolutionBasic(unsigned char *Img_in, char *M,unsigned char *Img_out,int Mask_Width,int rowImg,int colImg){
  __shared__ unsigned char imgT[TILE_WIDTH][TILE_WIDTH];

  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int row = by * TILE_WIDTH + ty;
  int col = bx * TILE_WIDTH + tx;
  
  unsigned int r = by*blockDim.y+ty;
  unsigned int c = bx*blockDim.x+tx;
  
  
   imgT[ty][tx] = Img_in[r*rowImg + c];

   __syncthreads();
   
    int N_start_point_i = tx- (Mask_Width/2);
    int N_start_point_j = ty - (Mask_Width/2);
    int Pvalue=0;
    for (int jj= 0;jj<Mask_Width;jj++) {
      for (int ii= 0;ii<Mask_Width;ii++) {
        if ((N_start_point_i+ii >= 0 && N_start_point_i + ii < TILE_WIDTH)&& (N_start_point_j+jj >= 0 && N_start_point_j + jj < TILE_WIDTH)) {
          Pvalue+=imgT[N_start_point_j+jj][N_start_point_i + ii]*M[ii*Mask_Width+jj];
        }
      }
      
    }
 __syncthreads();
 Img_out[r*rowImg + c]=conv(Pvalue);

}


int main(){


  int scale = 1;
  int delta = 0;
  int ddepth = CV_8UC1;

  clock_t start,end; 
  double elapsed_seconds;  

  Mat image;
  //Leer imagen en escala de grises
  image = imread("inputs/img6.jpg",0);
  Size s = image.size();
  int row=s.width;
  int col=s.height;
  char M[9] = {-1,0,1,-2,0,2,-1,0,1};
  //imwrite("./outputs/1089746672.png",image);
  
  //Separo memoria para las imagenes en el host
  int sizeM= sizeof(char)*9;
  int size = sizeof(unsigned char)*row*col;
  unsigned char *img=(unsigned char*)malloc(size);
  unsigned char *img_out=(unsigned char*)malloc(size);

  img=image.data;

  /////////////////////////SECUENCIAL///////////////////////////////////////////

  //cout<<gray_image.size().height<<gray_image.size().width<<endl;
  /// Generate grad_x and grad_y
  Mat grad_x, grad_y;
  /// Gradient X                  
  //   ( src  , grad_x, ddepth,dx,dy,scale,delta, BORDER_DEFAULT );
  start=clock();
  Sobel( image, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
  end= clock(); 
  /// Gradient Y
  //Sobel( image, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
  //imwrite("./outputs/1089746672.png",grad_x);

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
  char *d_M;
  cudaMalloc((void**)&d_img,size);
  cudaMalloc((void**)&d_img_out,size);
  cudaMalloc((void**)&d_M,sizeM);

  start=clock();
  cudaMemcpy(d_M,M,sizeM,cudaMemcpyHostToDevice);
  cudaMemcpy(d_img,img,size, cudaMemcpyHostToDevice);

  // Llamado al kernel
  KernelConvolutionBasic<<<dimGrid,dimBlock>>>(d_img,d_M,d_img_out,3,row,col);
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
  cudaFree(d_M);
  //cudaFree(d_sobelOutput);
  //free(img);
  //free(img_out);
    

  return 0; 
}
