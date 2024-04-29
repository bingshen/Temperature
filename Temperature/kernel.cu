#include "TemperatureMatrix.h"
#include "Draw.h"
#include <iostream>
#include <chrono>
#include <thread>

__device__ __host__ void init_temperature(TemperatureMatrix* mat)
{
    for(int i=0;i<PIC_SIZE;++i)
    {
        mat->pic[0][i]=MAX_TEMP;
        mat->pic[PIC_SIZE-1][i]=LOW_TEMP;
    }
}

__device__ __host__ void thomas(float a[],float b[],float c[],float d[],float x[])
{
    int n=PIC_SIZE;
    float c_prime[PIC_SIZE-1];
    float d_prime[PIC_SIZE];
    for(int i=0;i<PIC_SIZE;i++)
    {
        x[i]=0.0;
        d_prime[i]=0.0;
        if(i<PIC_SIZE-1)
            c_prime[i]=0.0;
    }
    c_prime[0]=c[0]/b[0];
    d_prime[0]=d[0]/b[0];
    for(int i=1;i<n-1;++i)
    {
        float temp=b[i]-a[i]*c_prime[i-1];
        c_prime[i]=c[i]/temp;
        d_prime[i]=(d[i]-a[i]*d_prime[i-1])/temp;
        d_prime[n-1]=(d[n-1]-a[n-1]*d_prime[n-2])/(b[n-1]-a[n-1]*c_prime[n-2]);
    }
    x[PIC_SIZE-1]=d_prime[PIC_SIZE-1];
    for(int i=PIC_SIZE-2;i>=0;--i)
        x[i]=d_prime[i]-c_prime[i]*x[i+1];
}

__global__ void kernel_x(TemperatureMatrix* mat,TemperatureMatrix* old_mat)
{
    int i=threadIdx.x+blockIdx.x*blockDim.x;
    if(i==0||i==PIC_SIZE-1)
        return;
    float u_temp[PIC_SIZE];
    for(int j=0;j<mat->nx;++j)
    {
        if(i>0 && i<PIC_SIZE-1)
            u_temp[j]=old_mat->pic[i][j]+old_mat->ay*(old_mat->pic[i+1][j]-2*old_mat->pic[i][j]+old_mat->pic[i-1][j]);
    }
    u_temp[0]=u_temp[1];
    u_temp[PIC_SIZE-1]=u_temp[PIC_SIZE-2];
    thomas(mat->A_y[0],mat->A_y[1],mat->A_y[2],u_temp,mat->pic[i]);
}

__global__ void kernel_y(TemperatureMatrix *mat,TemperatureMatrix* old_mat)
{
    int i=threadIdx.x+blockIdx.x*blockDim.x;
    if(i==0||i==PIC_SIZE-1)
        return;
    float u_temp[PIC_SIZE],u[PIC_SIZE];
    for(int j=0;j<mat->ny;++j)
    {
        if(i>0 && i<PIC_SIZE-1)
            u_temp[j]=old_mat->pic[j][i]+old_mat->ax*(old_mat->pic[j][i+1]-2*old_mat->pic[j][i]+old_mat->pic[j][i-1]);
    }
    u_temp[0]=MAX_TEMP;
    u_temp[PIC_SIZE-1]=LOW_TEMP;
    thomas(mat->A_x[0],mat->A_x[1],mat->A_x[2],u_temp,u);
    for(int j=0;j<mat->ny;++j)
        mat->pic[j][i]=u[j];
    init_temperature(mat);
}

__global__ void set_kernel(TemperatureMatrix* mat1,TemperatureMatrix* mat2)
{
    int i=threadIdx.x+blockIdx.x*blockDim.x;
    for(int j=0;j<PIC_SIZE;++j)
        mat1->pic[i][j]=mat2->pic[i][j];
}

int main()
{
    auto start=std::chrono::high_resolution_clock::now();
    TemperatureMatrix* mat=new TemperatureMatrix();
    TemperatureMatrix* old_mat=new TemperatureMatrix();
    init_temperature(mat);
    Draw::draw_pic(mat->pic,1);
    auto end=std::chrono::high_resolution_clock::now();
    auto duration=std::chrono::duration_cast<std::chrono::milliseconds>(end-start);
    cout<<1<<",cost:"<<duration.count()<<"ms"<<endl;
    for(int i=2;i<=100;i++)
    {
        auto start1=std::chrono::high_resolution_clock::now();
        for(int j=0;j<mat->steps;j++)
        {
            kernel_x<<<1,PIC_SIZE>>>(mat,old_mat);
            auto ret1=cudaDeviceSynchronize();
            set_kernel<<<1,PIC_SIZE>>>(old_mat,mat);
            auto ret2=cudaDeviceSynchronize();
            kernel_y<<<1,PIC_SIZE>>>(mat,old_mat);
            auto ret3=cudaDeviceSynchronize();
            set_kernel<<<1,PIC_SIZE>>>(old_mat,mat);
            auto ret4=cudaDeviceSynchronize();
        }
        Draw::draw_pic(mat->pic,i);
        auto end1=std::chrono::high_resolution_clock::now();
        auto duration1=std::chrono::duration_cast<std::chrono::milliseconds>(end1-start1);
        cout<<i-1<<",cost:"<<duration1.count()<<"ms"<<endl;
    }
    return 0;
}