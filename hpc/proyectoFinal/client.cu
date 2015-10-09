#include <czmq.h>
#include <iostream>
#include <cstring>

using namespace std;

#define PI 3.14159265

void print( float *U, float deltaX,int T,int N){
    for(int i=0; i<T; ++i){
        for(int j=0; j<N; ++j){
            cout<<j*deltaX<<" "<<U[i*N+j]<<endl;
        }
        cout<<endl;
    }
}

float f(float i, float deltax){
    float x = i*deltax;
    return 2*sin(2*PI*x); //initial condition
}

void handleMsg(zmsg_t* msg,float* U){
//answer:|idnodo|ti|N|Ui|
    string answer=zmsg_popstr(msg);
    int t=atoi(zmsg_popstr(msg));
    int N=atoi(zmsg_popstr(msg));
    for(int i=0;i<N;i++)
        U[t*N+i]=stod(zmsg_popstr(msg));
}

void esperarRespuesta(float* U,void* nodos,int T){
    //Recibir las soluciones, un mensaje por cada tiempo
    int time=1;
    zmq_pollitem_t items[] = {{nodos, 0, ZMQ_POLLIN, 0}};
    while (true) {
        zmq_poll(items, 1, 10 * ZMQ_POLL_MSEC);
        if (items[0].revents & ZMQ_POLLIN) {
            zmsg_t* msg=zmsg_recv(nodos);
            //zmsg_print(msg);
            zmsg_t* outmsg = zmsg_new();
            handleMsg(msg,U);
            time++;
            //Ya tengo todas las soluciones
            if(time==T)break;
        }
    }

}

//Secuencial
void FDSec(float *U, int T, int N,float r){
     for(int t=1; t<T; ++t){       //all nodes time
         U[t*N] = 0;  //Condicion de borde
         U[t*N+(N-1)] = 0; //Condicion de borde
        for(int pos=1; pos<N-1; ++pos){ //Sin tener en cuenta condiciones de borde
            U[t*N+pos] = r*U[(t-1)*N+(pos-1)] + U[(t-1)*N+pos]*(1-(2*r)) + r*U[(t-1)*N+(pos+1)];
        }
     }
}

int main(int argc, char** argv){
    // ./client ipcentral puerto
    zctx_t *context = zctx_new ();
    void* nodos= zsocket_new(context,ZMQ_DEALER);
    zsocket_connect(nodos, "tcp://%s:%s",argv[1],argv[2]);

    float *U,*U_sec,xa,xb,ta,tb,T,N,gamma,r;
    float deltaX,deltaT;
    int nodes,sizeU;

    clock_t start,end;
    double timeP,timeS;

    cin>>xa>>xb>>ta>>tb>>N>>T>>gamma;
    nodes = (N+2)*(T+2);
    sizeU = sizeof(float)*nodes;
    deltaX = (xb-xa) / float(N+1);
    deltaT = (tb-ta) / float(T+1);
    r = (gamma*deltaT)/(deltaX*deltaX);

    U = (float*)malloc(sizeU);
    U_sec=(float*)malloc(sizeU);
    if(r<=0  || r>=0.5 || deltaT>(deltaX*deltaX)/2.0){
        cout<<"r:"<<r<<"deltaT: "<<deltaT<<endl;
        cout<<"deltaX2:"<<(deltaX*deltaX)<<" (deltax2/2): "<<(deltaX*deltaX)/2<<endl;
        cout<<"Error !!!"<<endl;
        return 0;
    }

    for(int i=0; i<N+2; i++){
        U[i] = f(i,deltaX); //Matriz algoritmo paralelo
        U_sec[i]= f(i,deltaX);//Matriz algoritmo secuencial
    }


    //////////////////Algoritmo Paralelo//////////////////
    start=clock(); //Tiempo usando CUDA,OpenMP,CZMQ
    //task: |idcliente|task|r|T|N|U0|
    zmsg_t* msg=zmsg_new();
    zmsg_addstr(msg,"task");
    zmsg_addstr(msg,to_string(r).c_str());
    zmsg_addstr(msg,to_string(T+2).c_str());
    zmsg_addstr(msg,to_string(N+2).c_str());
    for(int i=0;i<N+2;i++)
        zmsg_addstr(msg,to_string(U[i]).c_str());
    //zmsg_print(msg);
    zmsg_send(&msg,nodos);

    //Recibo soluciones
    esperarRespuesta(U,nodos,T+2);
    end=clock();
    timeP=end-start;

    ///////////////Algoritmo Secuencial////////////////
    start=clock();
    FDSec(U_sec,T+2,N+2,r);
    end=clock();
    timeS=end-start;
    ///////////////////////////////////////////////////

    //Imprimo todas las soluciones

    // cout<<"Matriz Solución Algoritmo Paralelo"<<endl;
     print(U,deltaX,T+2,N+2);
    //cout<<"Matriz Solución Algoritmo Secuencial"<<endl;
    // print(U_sec,deltaX,T+2,N+2);
    //cout<<endl<<endl;

    cout<<"Tiempo Pararelo: "<< (timeP / CLOCKS_PER_SEC)<<endl;
    cout<<"Tiempo Secuencial: "<<(timeS/CLOCKS_PER_SEC)<<endl;
    cout<<"Aceleración obtenida: "<<(timeS/timeP)<<endl;
    return 0;
}
