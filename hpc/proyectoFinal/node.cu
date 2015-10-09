#include<bits/stdc++.h>
#include<cuda.h>

#define PI 3.14159265
#define BlockSize 1024

using namespace std;

__global__ void FD(float *U_d, int T, int N,float r){
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    for(int t=1; t<T; ++t){
        U_d[t*N]=0;
        U_d[t*N+(N-1)]=0;
        if(idx>0 && idx<N-1){
            U_d[t*N+idx] = r*U_d[(t-1)*N+(idx-1)]+(1-2*r)*U_d[(t-1)*N+idx]+r*U_d[(t-1)*N+(idx+1)];
        }
        __syncthreads();
    }

}

float f(float i, float deltax){
    float x = i*deltax;
    return 2*sin(2*PI*x); //initial condition
}

//print temperature for each node
void print( float *U, float deltaX,int T,int N){

    for(int i=0; i<T; ++i){
        for(int j=0; j<N; ++j){
            cout<<j*deltaX<<" "<<U[i*N+j]<<endl;
        }
        cout<<endl;
    }
}

int main(){
    float *U_d,*U_h, xa,xb,ta,tb,T,N,gamma,r;
    float deltaX,deltaT;
    int nodes,sizeU;
    cin>>xa>>xb>>ta>>tb>>N>>T>>gamma;
    nodes = (N+2)*(T+2);
    sizeU = sizeof(float)*nodes;

    U_h = (float*)malloc(sizeU);

    deltaX = (xb-xa) / float(N+1);
    deltaT = (tb-ta) / float(T+1);
    r = (gamma*deltaT)/(deltaX*deltaX);

    if(r<=0  || r>=0.5 || deltaT>(deltaX*deltaX)/2.0){
        cout<<"r:"<<r<<" k"<<deltaT<<endl;
        cout<<"error"<<endl;
        return 0;
    }

    for(int i=0; i<N+2; i++){
        U_h[i] = f(i,deltaX);
    }

    cudaMalloc((void**)&U_d,sizeU);
    cudaMemcpy(U_d,U_h,sizeU,cudaMemcpyHostToDevice);

    dim3 dimBlock(BlockSize);
    dim3 dimGrid(ceil((N+2)/float(BlockSize)));

    FD<<<dimGrid,dimBlock>>>(U_d,T+2,N+2,r);
    cudaMemcpy(U_h,U_d,sizeU,cudaMemcpyDeviceToHost);

    print(U_h,deltaX,T+2,N+2);
    cudaFree(U_d);


}
