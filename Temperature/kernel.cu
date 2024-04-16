#include "TemperatureMatrix.h"
#include "Draw.h"

__device__ __host__ void init_temperature(TemperatureMatrix* mat)
{
    for(int i=0;i<PIC_SIZE;++i)
    {
        mat->pic[0][i]=MAX_TEMP;
        mat->pic[PIC_SIZE-1][i]=LOW_TEMP;
    }
}

__device__ __host__ void tomas(float a[],float b[],float c[],float d[],float x[])
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

__global__ void kernel(TemperatureMatrix *mat)
{
    int i=threadIdx.x+blockIdx.x*blockDim.x;
    float u_temp[PIC_SIZE],u[PIC_SIZE];
    for(int j=0;j<mat->nx;++j)
    {
        u_temp[j]=mat->pic[i][j];
    }
    u_temp[0]=u_temp[1];
    u_temp[PIC_SIZE-1]=u_temp[PIC_SIZE-2];
    tomas(mat->A_y[0],mat->A_y[1],mat->A_y[2],u_temp,mat->pic[i]);
    for(int j=0;j<mat->ny;++j)
    {
        u_temp[j]=mat->pic[j][i];
    }
    tomas(mat->A_x[0],mat->A_x[1],mat->A_x[2],u_temp,u);
    for(int j=0;j<mat->ny;++j)
        mat->pic[j][i]=u[j];
    init_temperature(mat);
}

int main()
{
    TemperatureMatrix* mat=new TemperatureMatrix();
    init_temperature(mat);
    Draw::draw_pic(mat->pic,1);
    for(int i=2;i<=30;i++)
    {
        for(int j=0;j<mat->steps;j++)
        {
            kernel<<<1,PIC_SIZE>>>(mat);
            auto ret=cudaDeviceSynchronize();
        }
        Draw::draw_pic(mat->pic,i);
    }
    return 0;
}