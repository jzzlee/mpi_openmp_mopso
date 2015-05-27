#ifndef __COMMON_H_
#define __COMMON_H_
#include<cmath>

double distanceArray(double vec1[], double vec2[], int dim)
{
    double sum = 0;
	for(int n=0; n<dim; n++)
	    sum+= (vec1[n] - vec2[n])*(vec1[n] - vec2[n]);
	return sqrt(sum);
}

double distanceVector(double vec1[], double b0, double b1)
{
	return sqrt((vec1[0] - b0)*(vec1[0] - b0) + (vec1[1] - b1)*(vec1[1] - b1));
}

void minfastsort(double x[], int idx[], int n, int m)
{
    for(int i=0; i<m; i++)
	{
	    for(int j=i+1; j<n; j++)
			if(x[i]>x[j])
			{
			    double temp = x[i];
				x[i]        = x[j];
				x[j]        = temp;
				int id      = idx[i];
				idx[i]      = idx[j];
				idx[j]      = id;
			}
	}
}


#endif