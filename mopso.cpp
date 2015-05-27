#include<iostream>
#include "stdio.h"
#include "stdlib.h"
#include "mpi.h"
#include "omp.h"
#include"random.h"
#include"common.h"
#include"recombination.h"

using namespace std;




MPI_Status status;
//粒子
//每个粒子的位置和在各个目标上的适应值
struct Particle
{
	double x_var[numVariables];
	double y_obj[numObjectives];
	int rank;
	Particle();
	void rnd_init();
	void obj_eval();
	void operator=(const Particle &ind2);
};

//粒子初始化
Particle::Particle()
{
	for(int i=0; i<numVariables; i++)
		x_var[i] = 0.0;
	for(int n=0; n<numObjectives; n++)
        y_obj[n] = 0.0;
	rank = 0;
}
//对粒子位置随机初始化
void Particle::rnd_init()
{
    for(int n=0;n<numVariables;n++)
		x_var[n] = lowBound + rnd_uni(&rnd_uni_init)*(uppBound - lowBound);
}

//求目标值
void objectives(double x_var[], double y_obj[])
{
	double g = 0;
		for(int n=1;n<numVariables;n++)
			g+= x_var[n];
		g = 1 + 9*g/(numVariables-1);

		y_obj[0] = x_var[0];
		y_obj[1] = g*(1 - sqrt(x_var[0]/g) - x_var[0]*sin(10*pi*x_var[0])/g);
	

}
//计算粒子的适应值
void Particle::obj_eval()
{
    objectives(x_var,y_obj);
}
void Particle::operator=(const Particle &ind2)
{
	for(int i=0; i<numVariables; i++)
		x_var[i] = ind2.x_var[i];
	for(int n=0; n<numObjectives; n++)
		y_obj[n] = ind2.y_obj[n];
	rank  = ind2.rank;
}

struct TSOP
{
	Particle particle;
	double array[numObjectives];
	double namda[numObjectives];
};

//每个进程上的粒子群
TSOP *population = new TSOP[numThreads];
//粒子的邻居
double table[numNeighbors];
//在各个目标上的全局最优
double idealpoint[numObjectives];
//各个目标上取得最优时的位置
Particle indivpoint[numObjectives];
//局部最优
double gidealpoint[numObjectives];
Particle gindivpoint[numObjectives];

void moea(int my_rank)
{
	for(int n=0; n<numObjectives; n++) 
	{
		gidealpoint[n] = 1.0e+30; 
	}
	if(my_rank==0)
		for(int n=0; n<numObjectives; n++) 
		{
			idealpoint[n] = 1.0e+30;

		}
}

void init_uniformweight(int sd, int my_rank)
{
	if(numObjectives==2)
	{
		int my_thread = omp_get_thread_num();
		population[my_thread].array[0] = my_thread + my_rank*numThreads;
		population[my_thread].array[1] = sd - population[my_thread].array[0];
		double lambda;
	    for(int j=0; j<numObjectives; j++)
		{
			lambda = 1.0*population[my_thread].array[j]/sd;
			population[my_thread].namda[j] = lambda;
		}
	}
	else if(numObjectives==3)
	{
		int my_thread = omp_get_thread_num();
		population[my_thread].array[0] = my_thread + my_rank*numThreads;
		population[my_thread].array[1] = sd - population[my_thread].array[0];
		double lambda;
	    for(int j=0; j<numObjectives; j++)
		{
			lambda = 1.0*population[my_thread].array[j]/sd;
			population[my_thread].namda[j] = lambda;
		}
	}

}

void update_gbest(Particle &indiv)
{
	for(int n=0; n<numObjectives; n++)    
	{
		if(indiv.y_obj[n]<gidealpoint[n])
		{
			gidealpoint[n]  = indiv.y_obj[n];
			gindivpoint[n]  = indiv;
		}
	}
}


double gideal_points[numCores*numObjectives];
double x_tmp0[numCores*numVariables];
double x_tmp1[numCores*numVariables];
void update_reference(int my_rank)
{
/*	
	int arrayblocks[3] = {numObjectives, numVariables, numVariables};
	MPI_Aint arraydisplaces[3];
	MPI_Aint gideal_addr, gindiv0_addr, gindiv1_addr;
	MPI_Get_address(&gidealpoint, &gideal_addr);
	MPI_Get_address(&(gindivpoint[0].x_var), &gindiv0_addr);
	MPI_Get_address(&(gindivpoint[1].x_var), &gindiv1_addr);
	arraydisplaces[0] = 0;
	arraydisplaces[1] = gindiv0_addr - gideal_addr;
	arraydisplaces[2] = gindiv1_addr - gideal_addr;
	MPI_Datatype arraytypes[3] = {MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE};
	MPI_Datatype idealtype;
	MPI_Type_create_struct(3, arrayblocks, arraydisplaces, arraytypes, &idealtype);
	MPI_Type_commit(&idealtype);

	if(my_rank !=0 )
	{
		MPI_Gather(gidealpoint, numObjectives, idealtype, gideal_points, numObjectives, idealtype, 0, MPI_COMM_WORLD);
	}
	else
	{
		MPI_Gather(gidealpoint, numObjectives, idealtype, gideal_points, numObjectives, idealtype, 0, MPI_COMM_WORLD);

		//find the best idealpoint
		int index[numObjectives] = {0,0};
		for(int i=0;i<numCores*numObjectives;i+=2)
		{
			if(y_tmp[i]<idealpoint[0])
			{
				idealpoint[0] = y_tmp[i];
				index[0] = i/numObjectives;
			}
		}
		for(int i=1;i<numCores*numObjectives;i+=2)
		{
			if(y_tmp[i]<idealpoint[1])
			{
				idealpoint[1] = y_tmp[i];
				index[1] = i/numObjectives;
			}
		}
		//update the reference
		for(int i=0;i!=numVariables;++i)
		{
			indivpoint[0].x_var[i] = x_tmp[numVariables*index[0] + i];
			indivpoint[1].x_var[i] = x_tmp[numVariables*index[1] + i];
		}
		for(int i=0;i!=numObjectives;++i)
		{
			indivpoint[0].y_obj[i] = y_tmp[numObjectives*index[0] + i];
			indivpoint[1].y_obj[i] = x_tmp[numObjectives*index[1] + i];
		}
		
	}
*/
	if(my_rank !=0 )
	{
		MPI_Gather(gidealpoint, numObjectives, MPI_DOUBLE, gideal_points, numObjectives, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Gather(gindivpoint[0].x_var, numVariables, MPI_DOUBLE, x_tmp0, numVariables, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Gather(gindivpoint[1].x_var, numVariables, MPI_DOUBLE, x_tmp1, numVariables, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	}
	else
	{
		MPI_Gather(gidealpoint, numObjectives, MPI_DOUBLE, gideal_points, numObjectives, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Gather(gindivpoint[0].x_var, numVariables, MPI_DOUBLE, x_tmp0, numVariables, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Gather(gindivpoint[1].x_var, numVariables, MPI_DOUBLE, x_tmp1, numVariables, MPI_DOUBLE, 0, MPI_COMM_WORLD);

		//find the best idealpoint
		int index[numObjectives] = {0,0};
		for(int i=0;i<numCores*numObjectives;i+=numObjectives)
		{
			if(gideal_points[i]<idealpoint[0])
			{
				idealpoint[0] = gideal_points[i];
				index[0] = i/numObjectives;
			}
		}
		for(int i=1;i<numCores*numObjectives;i+=numObjectives)
		{
			if(gideal_points[i]<idealpoint[1])
			{
				idealpoint[1] = gideal_points[i];
				index[1] = i/numObjectives;
			}
		}
		//update the indivpoint
		for(int i=0;i!=numVariables;++i)
		{
			indivpoint[0].x_var[i] = x_tmp0[numVariables*index[0] + i];
			indivpoint[1].x_var[i] = x_tmp1[numVariables*index[1] + i];
		}
	}
	MPI_Bcast(idealpoint, numObjectives, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(indivpoint[0].x_var, numVariables, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(indivpoint[1].x_var, numVariables, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	for(int i=0;i!=numObjectives;++i)
		indivpoint[i].obj_eval();
}


void init_population(int my_rank)
{
	int my_thread = omp_get_thread_num();
	population[my_thread].particle.rnd_init();
	population[my_thread].particle.obj_eval();
	
}

int two_neighbors[numCores*2];
Particle *child = new Particle[numThreads];
//进化
void bound(int my_thread)
{
	for(int i=0;i!=numVariables;++i)
	{
		if(child[my_thread].x_var[i]<lowBound)
			child[my_thread].x_var[i] = lowBound;
		else if(child[my_thread].x_var[i]>uppBound)
			child[my_thread].x_var[i] = uppBound;
	}
}

Particle exchange_particle0, exchange_particle1;
double particle0_xvar[numVariables*numCores], particle1_xvar[numVariables*numCores];
double tmp0_xvar[numVariables*numCores], tmp1_xvar[numVariables*numCores];
void exchange(int gen, int my_rank)
{
	MPI_Gather(population[numThreads-1].particle.x_var, numVariables, MPI_DOUBLE, tmp0_xvar, numVariables, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Gather(population[0].particle.x_var, numVariables, MPI_DOUBLE, tmp1_xvar, numVariables, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	for(int i=0,j=numVariables;j<numVariables*numCores;++j,++i)
	{
		particle0_xvar[j] = tmp0_xvar[i];
		particle1_xvar[i] = tmp1_xvar[j];
	}
	
	MPI_Scatter(particle0_xvar, numVariables, MPI_DOUBLE, exchange_particle0.x_var, numVariables, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Scatter(particle1_xvar, numVariables, MPI_DOUBLE, exchange_particle1.x_var, numVariables, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	if(my_rank==0)
		for(int i=0;i!=numVariables;++i)
				exchange_particle0.x_var[i] = population[0].particle.x_var[i];
	if(my_rank==numCores-1)
		for(int i=0;i!=numVariables;++i)
				exchange_particle1.x_var[i] = population[numThreads-1].particle.x_var[i];
}

void evolution(int my_rank)
{
	int my_thread = omp_get_thread_num();
	int r1 = int(numExchange*rnd_uni(&rnd_uni_init));
	int r3 = int(numNeighbors*rnd_uni(&rnd_uni_init));
	int r2 = int(numNeighbors*rnd_uni(&rnd_uni_init));
	
	Particle child2;
	double rand = rnd_uni(&rnd_uni_init);
	int half_neighbors = numNeighbors/2;
	if(my_thread > half_neighbors)
	{
		if(rand < 1 - 1.0*half_neighbors/my_thread)
			realbinarycrossover(exchange_particle1, population[r2].particle, child[my_thread], child2);
		else
			realbinarycrossover(population[r3].particle, population[r2].particle, child[my_thread], child2);
	}
	else if(my_thread < half_neighbors)
	{
		if(rand < 1.0*(half_neighbors-my_thread)/(numNeighbors - my_thread))
			realbinarycrossover(exchange_particle0, population[r2].particle, child[my_thread], child2);
		else
			realbinarycrossover(population[r3].particle, population[r2].particle, child[my_thread], child2);
	}
	else
		realbinarycrossover(population[r3].particle, population[r2].particle, child[my_thread], child2);
	/*
	if(rand < 0.2)
		if(r1)
			realbinarycrossover(exchange_particle1, population[r2].particle, child[my_thread], child2);
		else
			realbinarycrossover(exchange_particle0, population[r2].particle, child[my_thread], child2);
	else
		realbinarycrossover(population[r3].particle, population[r2].particle, child[my_thread], child2);
	*/
	realmutation(child[my_thread], 1.0/numVariables);
	child[my_thread].obj_eval();
}

char    strFunctionType[256];
double scalar_func(double y_obj[], double namda[], Particle *gindiv)
{
	double fvalue = 0;
    
	// Tchebycheff approach
	if(!strcmp(strFunctionType,"_TCH1"))
	{
		double max_fun = -1.0e+30;
		for(int n=0; n<numObjectives; n++)
		{
			double diff = fabs(y_obj[n] - idealpoint[n] );
			double feval;
			if(namda[n]==0) 
				feval = 0.00001*diff;
			else
			    feval = diff*namda[n];
			if(feval>max_fun) max_fun = feval;

		}
		
		fvalue = max_fun;
	}

	// normalized Tchebycheff approach
	if(!strcmp(strFunctionType,"_TCH2"))
	{
		double scale[numObjectives];
		for(int i=0; i<numObjectives; i++)
		{
			double min = 1.0e+30, max = -1.0e+30;
			for(int j=0; j<numObjectives; j++)
			{
				double tp = gindiv[j].y_obj[i];
				if(tp>max) max = tp;
				if(tp<min) min = tp;
			}
			scale[i] = max-min;
			if(max-min==0) return 1.0e+30;
		}

		double max_fun = -1.0e+30;
		for(int n=0; n<numObjectives; n++)
		{
			double diff = (y_obj[n] - idealpoint[n])/scale[n];
			double feval;
			if(namda[n]==0) 
				feval = 0.0001*diff;
			else
			    feval = diff*namda[n];
			if(feval>max_fun) max_fun = feval;

		}
		fvalue = max_fun;
	}

/*
	//* Boundary intersection approach
	if(!strcmp(strFunctionType,"_PBI"))
	{

		// normalize the weight vector (line segment)
		double nd = norm_vector(namda);
		for(int i=0; i<numObjectives; i++)
			namda[i] = namda[i]/nd;

		vector <double> realA(numObjectives);
		vector <double> realB(numObjectives);

		// difference beween current point and reference point
		for(int n=0; n<numObjectives; n++)
			realA[n] = (y_obj[n] - idealpoint[n]);

		// distance along the line segment
		double d1 = fabs(innerproduct(realA,namda));

		// distance to the line segment
		for(int n=0; n<numObjectives; n++)
			realB[n] = (y_obj[n] - (idealpoint[n] + d1*namda[n]));
		double d2 = norm_vector(realB);

		fvalue = d1 + 5*d2;

	}
	*/
	return fvalue;
}
void update_problem(int my_rank)
{
	int my_thread = omp_get_thread_num();
	double f1, f2;
	f1 = scalar_func(population[my_thread].particle.y_obj, population[my_thread].namda, indivpoint);
	f2 = scalar_func(child[my_thread].y_obj, population[my_thread].namda, indivpoint);
	if(f2<f1)
		population[my_thread].particle = child[my_thread];
}

char *filename="mpifile_1.txt";
const int local_length = numThreads*numObjectives;
const int global_length = numCores*local_length;
double globalpareto[global_length];
double localpareto[local_length];

const int local_lengthx = numThreads*numVariables;
const int global_lengthx = numCores*local_lengthx;
double globalparetox[global_lengthx];
double localparetox[local_lengthx];

void save_localpareto()
{
	int my_thread = omp_get_thread_num();
	for(int i=0;i<numObjectives;++i)
		localpareto[my_thread*numObjectives+i]=population[my_thread].particle.y_obj[i];
	for(int j=0;j<numVariables;++j)
		localparetox[my_thread*numVariables+j]=population[my_thread].particle.x_var[j];
}

void save_pareto(int my_rank)
{
	MPI_Gather(localpareto, local_length, MPI_DOUBLE, globalpareto, local_length, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Gather(localparetox, local_lengthx, MPI_DOUBLE, globalparetox, local_lengthx, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	if(my_rank==0)
	{
		ofstream ff(filename);
		for(int i=1;i<=global_length;++i)
		{
			if(!(i%numObjectives))
				ff<<globalpareto[i-1]<<endl;
			else
				ff<<globalpareto[i-1]<<"\t";
		}
		ff.close();
/*		ofstream ffx(filename);
		for(int j=1;j<=global_lengthx;++j)
		{
			if(!(j%numVariables))
				ffx<<globalparetox[j-1]<<endl;
			else
				ffx<<globalparetox[j-1]<<" ";
		}
		ff.close();
		*/
	}
}


char *filename1="child.txt";
const int local_ch_length = numThreads*numObjectives;
const int global_ch_length = numCores*local_ch_length;
double globalchild[global_ch_length];
double localchild[local_ch_length];

const int local_chx_length = numThreads*numVariables;
const int global_chx_length = numCores*local_chx_length;
double globalchildx[global_chx_length];
double localchildx[local_chx_length];
void save_localchild()
{
	int my_thread = omp_get_thread_num();
	for(int i=0;i<numObjectives;++i)
		localchild[my_thread*numObjectives+i]=child[my_thread].y_obj[i];
	for(int j=0;j<numVariables;++j)
		localchildx[my_thread*numVariables+j]=child[my_thread].x_var[j];
}

void save_child(int my_rank)
{
	MPI_Gather(localchild, local_ch_length, MPI_DOUBLE, globalchild, local_ch_length, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Gather(localchildx, local_chx_length, MPI_DOUBLE, globalchildx, local_chx_length, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	if(my_rank==0)
	{
		ofstream fff(filename1);
		for(int i=0;i<100;++i)
		{
			for(int j=0;j!=numObjectives;++j)
				fff<<globalchild[i*numObjectives+j]<<" ";
			for(int j=0;j!=numVariables;++j)
				fff<<globalchildx[i*numVariables+j]<<" ";
			fff<<endl;
		}
		fff.close();
	}
/*	if(my_rank==0)
	{
		for(int i=0;i!=25;++i)
		{
			for(int j=0;j!=30;++j)
				cout<<localchildx[i*numVariables +j]<<" ";
			cout<<endl;
		}
	}*/
}

int main()
{
	

	strcpy(strFunctionType,"_TCH2");  
	int my_rank, comm_sz;
	MPI_Init(0,0);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
	
	MPI_Barrier(MPI_COMM_WORLD);
	double start,end;
	start = MPI_Wtime();

	seed = (seed + 111+ 123*my_rank)%1235 ;
	rnd_uni_init = -(long)seed;

	moea(my_rank);
#	pragma omp parallel num_threads(numThreads)
	{
		init_uniformweight(numParticles-1, my_rank);
		init_population(my_rank);
	}
	
	
	for(int i=0;i!=numThreads;++i)
		update_gbest(population[i].particle);

	MPI_Barrier(MPI_COMM_WORLD);
	
	update_reference(my_rank);
	
	for(int gen=2; gen<=mg; gen++)
	{
		MPI_Barrier(MPI_COMM_WORLD);
		exchange(gen, my_rank);

//		MPI_Barrier(MPI_COMM_WORLD);
		#pragma omp parallel num_threads(numThreads)
		evolution(my_rank);

		

		for(int i=0;i!=numThreads;++i)
			update_gbest(child[i]);

		MPI_Barrier(MPI_COMM_WORLD);
		update_reference(my_rank);
		# pragma omp parallel num_threads(numThreads)
		update_problem(my_rank);
		
	}
//	#pragma omp parallel num_threads(numThreads)
	//save_localchild();

//	save_child(my_rank);

#pragma omp parallel num_threads(numThreads)
	save_localpareto();

	save_pareto(my_rank);

	MPI_Barrier(MPI_COMM_WORLD);
	end = MPI_Wtime();
	double local_time = 1000*(end - start);
	double real_time;
	MPI_Reduce(&local_time, &real_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	if(my_rank==0)
		cout<<real_time<<endl;



	
	MPI_Finalize();
	
//	int a;
	//cin>>a;
	
	return 0;
}
