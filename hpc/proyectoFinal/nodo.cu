#include<bits/stdc++.h>
#include<czmq.h>
#include<cuda.h>
#include<omp.h>

#define BlockSize 1024

using namespace std;

int respuestas=0;
string dbg = "------------------------";
float *Ui;     //Ui:Solución de un tiempo i
zframe_t* cliente;

struct info{
    int T;
    int ti=1;
    int N;
    int respuestas;
};

unordered_map<string, float* > clients;
unordered_map<string, info> state; //pair<respuestas, Tiempos>
vector<zframe_t *> nodes;

//N: offsetG de U_prev_d
__global__ void FD(float *U_prev_d, float *U_new_d, int init, int N,float r, int offsetL){
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    //int i = idx+init+initG;
    int i=init+idx+1;
    if(idx<offsetL)
        U_new_d[idx] = r*U_prev_d[i-1]+(1-2*r)*U_prev_d[i]+r*U_prev_d[i+1];
}


// N is length of U_prev
// offsetL is length of Ui_next
void launchKernel(float* Ui_prev,float *Ui_next,int offsetL, int tid, int N, float  r, int initG){
    cudaSetDevice(tid);
    int init;
    float *U_new_d,*U_prev_d;
    init = offsetL * tid;
    cudaMalloc((void**)&U_prev_d,sizeof(float)*N);
    cudaMalloc((void**)&U_new_d,sizeof(float)*offsetL);
    cudaMemcpy(U_prev_d,Ui_prev, sizeof(float)*N, cudaMemcpyHostToDevice);
    dim3 dimBlock(BlockSize);
    dim3 dimGrid(ceil(offsetL/float(BlockSize)));
    FD<<<dimGrid, dimBlock>>>(U_prev_d,U_new_d,init,N,r,offsetL);
    cudaDeviceSynchronize();
    cudaMemcpy(&Ui_next[init],U_new_d,sizeof(float)*offsetL,cudaMemcpyDeviceToHost);
    cudaFree(U_new_d);
    cudaFree(U_prev_d);
}

// U_i: parte de la matriz U en un tiempo especifico a resolver
// n: Numero nodos de U_prev, longitud U_prev
void paralelizar(float *Ui_prev,float* Ui_next,int initG,int offsetL,int n, float r){
    int tid;
#pragma omp parallel  private(tid) shared(Ui_next)
    {
        tid = omp_get_thread_num();
        //cout<<tid<<endl;
        launchKernel(Ui_prev,Ui_next,offsetL, tid,n,r, initG);
    }
    cudaDeviceSynchronize();
}



/*
  Tipos de mensajes.
  task: |idcliente|task|r|T|N|U0| client->main
  register: |idnodo|register| node-> main
  solve: |idnodo|idnodo|solve|id_client|offsetG|U_prev_d|initG|N|r| main->node
  answer_i: |idnodo|answer_i| id_client|initG|offsetL|Ui_new|r| node->main
  answer:|idnodo|ti|N|Ui| main->client
*/

//nodos: socket, nodes:vector
//EL tamaño del vector debe ser multiplo del numero de nodos
void distribuir(void* nodos, float *Ui_next, zframe_t* id_client, float* U_prev, int N,float r){
    int gpus;
    int numNodos = nodes.size()+1;
    int offsetG = N/numNodos;
    string idc = zframe_strhex(id_client);

    cudaGetDeviceCount(&gpus);
    omp_set_num_threads(gpus);
    int offsetL=offsetG/gpus;
    float* U_preInterno=(float*)malloc(sizeof(float)*(offsetG+2));
    U_preInterno[0]=0.0;
    memcpy(&U_preInterno[1],&U_prev,sizeof(float)*(offsetG+1));
    paralelizar(U_preInterno,Ui_next,0,offsetL,offsetG+2,r);

    //cout<<offsetG<<"Voy a distribuir"<<endl;
    for(int i=1;i<numNodos;i++){
        zmsg_t* msg = zmsg_new();
        zframe_t* copyNode = zframe_dup(nodes[i-1]);
        zframe_t* copyIdC = zframe_dup(id_client);
        zframe_t* cpyNode = zframe_dup(nodes[i-1]);
        zmsg_prepend(msg,&copyNode);
        zmsg_append(msg,&cpyNode); //important
        zmsg_addstr(msg,"solve");
        zmsg_append(msg,&copyIdC);
        zmsg_addstr(msg,to_string(offsetG).c_str());
        int pos = i*offsetG;

        //int limit = (i==numNodos-1)? offsetG : offsetG+1;
        int j;
        for(j=-1;j<=offsetG;j++)
            zmsg_addstr(msg,to_string(U_prev[pos+j]).c_str());

        cout<<"pos+j"<<pos<<" N-1"<<N-1<<endl;
        if(pos+j<N-1)
            zmsg_addstr(msg,to_string(U_prev[pos+j]).c_str());
        else if(pos+j==N-1)
            zmsg_addstr(msg,to_string(0).c_str());

        zmsg_addstr(msg,to_string(i*offsetG).c_str());
        zmsg_addstr(msg,to_string(N).c_str());
        zmsg_addstr(msg,to_string(r).c_str());
        cout<<dbg<<endl;
        zmsg_print(msg);
        zmsg_send(&msg,nodos);
        cout<<dbg<<endl;
        state[idc].respuestas++;
        //cout<<"respuestas"<< state[idc].respuestas<<endl;
    }
}

void handleNodeMessage(zmsg_t* msg, void *nodos){
    zframe_t* idframe = zmsg_pop(msg);
    string idNode = zframe_strhex(zframe_dup(idframe));
    string code = zmsg_popstr(msg);

    if(code.compare("register")==0){
        nodes.push_back(idframe);
    }else if(code.compare("task")==0){
        float r = stod(zmsg_popstr(msg));
        int T = atoi(zmsg_popstr(msg));
        int N = atoi(zmsg_popstr(msg));
        float *U0 = (float*)malloc(sizeof(float)*N);
        info x;
        x.T = T;
        x.N = N;
        x.ti = 1;
        x.respuestas = 0;
        state[idNode] = x;
        clients[idNode] = (float*)malloc(sizeof(float)*N);
        for(int i=0;i<N;i++)
            U0[i] = stod(zmsg_popstr(msg));
        distribuir(nodos,clients[idNode],idframe,U0,N,r);
    }else if(code.compare("solve")==0){
        zframe_t *id_client = zmsg_pop(msg);
        int offsetG = atoi(zmsg_popstr(msg));
        float* U_prev = (float*)malloc(sizeof(float)*(offsetG+2));
        for(int i=0;i<offsetG+2;i++)
            U_prev[i]=stod(zmsg_popstr(msg));
        int initG = atoi(zmsg_popstr(msg));
        int N = atoi(zmsg_popstr(msg));
        float r = stod(zmsg_popstr(msg));
        float *Ui_next = (float*)malloc(sizeof(float)*offsetG);
        int gpus;
        cudaGetDeviceCount(&gpus);
        omp_set_num_threads(gpus);
        int offsetL=offsetG/gpus;
        paralelizar(U_prev,Ui_next,initG,offsetL,offsetG+2,r);

        //offsetL: el tamaño de Ui_new calculado
        // answer_i: |idnodo|answer_i| id_client|initG|offsetG|Ui_new|r| node->main

        zmsg_t* msg_reply=zmsg_new();
        zmsg_addstr(msg_reply,"answer_i");
        //zframe_print(id_client,"id cliente");
        zmsg_append(msg_reply,&id_client);
        zmsg_addstr(msg_reply,to_string(initG).c_str());
        zmsg_addstr(msg_reply,to_string(offsetG).c_str());
        for(int i=0;i<offsetG;i++)
            zmsg_addstr(msg_reply,to_string(Ui_next[i]).c_str());
        zmsg_addstr(msg_reply,to_string(r).c_str());
        cout<<dbg<<endl;
        zmsg_print(msg_reply);
        zmsg_send(&msg_reply,nodos);
    }else if(code.compare("answer_i")==0){
        zframe_t *idclient = zmsg_pop(msg);
        string idc = zframe_strhex(idclient);
        int initG=atoi(zmsg_popstr(msg));
        int offsetG = atoi(zmsg_popstr(msg)); // Ni

        //Escribo la solución en el vector de solucion del tiempo i
        //cout<<"offsetL"<<offsetL<<endl;
        for(int i=0;i<offsetG;i++){
           clients[idc][initG+i] = stod(zmsg_popstr(msg));
        }
        //cout<<"respuestas"<<state[idc].respuestas<<endl;
        //zframe_print(id,"idC");
        state[idc].respuestas--;
        float r=stod(zmsg_popstr(msg));

        if(state[idc].respuestas==0){
            //answer:|idnodo|answer|ti|N|Ui|
            zmsg_t* msg_reply = zmsg_new();
            zframe_t* copyIDc = zframe_dup(idclient);
            zmsg_append(msg_reply,&copyIDc);
            zmsg_addstr(msg_reply,"answer");
            zmsg_addstr(msg_reply,to_string(state[idc].ti).c_str());
            zmsg_addstr(msg_reply,to_string(state[idc].N).c_str());
            clients[idc][0] = 0;
            clients[idc][state[idc].N-1] = 0;
            for(int i=0;i<state[idc].N;i++){
                zmsg_addstr(msg_reply,to_string(clients[idc][i]).c_str());
            }
            cout<<dbg<<endl;
            zmsg_print(msg_reply);
            zmsg_send(&msg_reply,nodos);
            state[idc].respuestas = 0;
            state[idc].ti++;
            if(state[idc].ti<=state[idc].T)
                distribuir(nodos,clients[idc],idclient,clients[idc],state[idc].N,r);
        }
    }
}

int main(int argc, char** argv){
    zctx_t *context = zctx_new ();
    //./nodo ipcentral puerto numNodos
    void* nodos;
    int numNodos=atoi(argv[3]);
    if(numNodos!=-1){ //Nodo principal
        nodos = zsocket_new(context,ZMQ_ROUTER);
        zsocket_bind(nodos, "tcp://*:%s",argv[2]);
    }else{
        zmsg_t *msg = zmsg_new();
        zmsg_addstr(msg,"register");
        nodos = zsocket_new(context,ZMQ_DEALER);
        zsocket_connect(nodos, "tcp://%s:%s",argv[1],argv[2]);

        zmsg_send(&msg,nodos);
    }
    //Escuchar los tipos de mensajes
    zmq_pollitem_t items[] = {{nodos, 0, ZMQ_POLLIN, 0}};
    while (true) {
        zmq_poll(items, 1, 10 * ZMQ_POLL_MSEC);
        if (items[0].revents & ZMQ_POLLIN) {
            cout<<"______________________________"<<endl;
            zmsg_t* msg = zmsg_recv(nodos);
            zmsg_print(msg);
            cout<<"______________________________"<<endl;
            handleNodeMessage(msg,nodos);
        }
    }

    return 0;
}
