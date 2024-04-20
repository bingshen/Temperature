#pragma once
#include "ShareMemoryManage.h"

#define PIC_SIZE 512
#define MAX_TEMP 100
#define LOW_TEMP 0

class TemperatureMatrix:public ShareMemoryManage
{
public:
	int nx;
	int ny;
	float dx;
	float dy;
	float dt;
	float alpha;
	int steps;
	float pic[PIC_SIZE][PIC_SIZE];
	float A_x[3][PIC_SIZE];
	float A_y[3][PIC_SIZE];
	float ax,ay;

	void set(TemperatureMatrix* mat)
	{
		for(int i=0;i<PIC_SIZE;++i)
			for(int j=0;j<PIC_SIZE;++j)
				this->pic[i][j]=mat->pic[i][j];
	}

	TemperatureMatrix()
	{
		this->nx=PIC_SIZE;
		this->ny=PIC_SIZE;
		this->dx=0.1;
		this->dy=0.1;
		this->dt=0.01;
		this->alpha=0.1;
		this->steps=200;
		for(int i=0;i<PIC_SIZE;++i)
			for(int j=0;j<PIC_SIZE;++j)
				this->pic[i][j]=0.0;
		this->ax=alpha*dt/(dx*dx);
		this->ay=alpha*dt/(dy*dy);
		for(int i=0;i<3;i++)
			for(int j=0;j<PIC_SIZE;++j)
			{
				this->A_x[i][j]=0.0;
				this->A_y[i][j]=0.0;
			}
		for(int i=0;i<PIC_SIZE;++i)
		{
			this->A_x[0][i]=-ax;
			this->A_x[1][i]=1+2*ax;
			this->A_x[2][i]=-ax;
			this->A_y[0][i]=-ay;
			this->A_y[1][i]=1+2*ay;
			this->A_y[2][i]=-ay;
		}
		this->A_x[0][0]=0.0;
		this->A_x[2][PIC_SIZE-1]=0.0;
		this->A_y[0][0]=0.0;
		this->A_y[2][PIC_SIZE-1]=0.0;
	}
};