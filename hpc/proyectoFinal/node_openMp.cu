#include<bits/stdc++.h>
#include<cuda.h>
#include<omp.h>

#define PI 3.14159265
#define BlockSize 1024

using namespace std;

__global__ void FD(float *U_prev_d, float *U_new_d, int init, int N,float r){
   int idx = blockIdx.x*blockDim.x + threadIdx.x;
   int i = idx+init;
   if(i>0 && i<N-1)
     U_new_d[idx] = r*U_prev_d[i-1]+(1-2*r)*U_prev_d[i]+r*U_prev_d[i+1];

}

float f(float i, float deltax){
    float x = i*deltax;
    return 2*sin(2*PI*x); //initial condition
}
//print temperature for each node
void print( float *U, float deltaX,int T,int N){
    for(int i=0; i<3; ++i){
        for(int j=0; j<N; ++j){
            cout<<j*deltaX<<" "<<U[i*N+j]<<endl;
        }
        cout<<endl;
    }
}
// N is length of U_prev
void launchKernel(float* U, int offset, int tid, int t, int N, float  r){
    cudaSetDevice(tid);
    int init, end,pos, pos2;
    float *U_new_d,*U_prev_d;
    init = offset * tid;
    end = init + offset;
    pos = (t-1)*N;
    pos2 = (t*N)+init;
    cudaMalloc((void**)&U_prev_d,sizeof(float)*N);
    cudaMalloc((void**)&U_new_d,sizeof(float)*offset);
    //cudaMemcpy(U_prev_d, &U[0], sizeof(float)*N, cudaMemcpyHostToDevice);
    cudaMemcpy(U_prev_d, &U[pos], sizeof(float)*N, cudaMemcpyHostToDevice);
    //pos = t*N+init;
    dim3 dimBlock(BlockSize);
    dim3 dimGrid(ceil(offset/float(BlockSize)));
    FD<<<dimGrid, dimBlock>>>(U_prev_d,U_new_d,init,N,r);
    cudaDeviceSynchronize();
    cudaMemcpy(&U[pos2],U_new_d,sizeof(float)*offset,cudaMemcpyDeviceToHost);
    cudaFree(U_new_d);
    cudaFree(U_prev_d);

}

int main(){
    float *U_d,*U_h, xa,xb,ta,tb,T,N,gamma,r,*tmp;
    float deltaX,deltaT;
    int nodes,sizeU,gpus,offset,tid;
    cudaGetDeviceCount(&gpus);
    omp_set_num_threads(gpus);
    cout<<"gpus:"<<gpus<<endl;

    cin>>xa>>xb>>ta>>tb>>N>>T>>gamma;
    nodes = (N+2)*(T+2);
    sizeU = sizeof(float)*nodes;
    deltaX = (xb-xa) / float(N+1);
    deltaT = (tb-ta) / float(T+1);
    r = (gamma*deltaT)/(deltaX*deltaX);
    offset = ceil((N+2)/float(gpus));

    U_h = (float*)malloc(sizeU);
    tmp = (float*)malloc(N+2);

    if(r<=0  || r>=0.5 || deltaT>(deltaX*deltaX)/2.0){
        cout<<"r:"<<r<<" k"<<deltaT<<endl;
        cout<<"error"<<endl;
        return 0;
    }
    for(int i=0; i<N+2; i++){
        U_h[i] = f(i,deltaX);
    }
    for(int t=1; t<T+1; ++t){
#pragma omp parallel  private(tid) shared(U_h)
        {
            tid = omp_get_thread_num();
            //cout<<tid<<endl;
            launchKernel(U_h, offset, tid,t,N+2,r);
            
        }
        U_h[t*int(N+2)]=0;                                                                                                   
        U_h[t*int(N+2)+int(N+1)]=0;   
        cudaDeviceSynchronize(); 
    }

print(U_h,deltaX,T+2,N+2);
}
