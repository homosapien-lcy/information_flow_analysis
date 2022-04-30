//this program calculates information flow from A and A inverse outputed by matlab
using namespace std;
#include <stdio.h>
#include <math.h>
#include <iostream>
#include <time.h>

#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse.h>
#include <iomanip>

//fill in the number below according to the output from the matlab preprocessing script
static const unsigned int m_size = 12865;
static const unsigned int element_num = 686082;
static const unsigned int LUT_size = 673217;
static const unsigned int number_of_rounds = 10;

static const unsigned int element_load_max = 268435456; //2048MB / 8B
const unsigned int col_load_max = floor((double) element_load_max / (double) m_size);
const unsigned int number_of_loads = ceil((double) m_size / (double) col_load_max);

//print
void print_array(double* array, unsigned int amount)
{
	unsigned int i;

	for(i = 0; i < amount; i++)
	{
		printf("Element %d = %f \n", i, array[i]);
	}
}

void print_array(int* array, unsigned int amount)
{
	unsigned int i;

	for(i = 0; i < amount; i++)
	{
		printf("Element %d = %d \n", i, array[i]);
	}
}

//read matrix
void read_matrix(const char *filename, double *matrix, unsigned int amount)
{
	char bytes[8];
	FILE *file_pt;
	unsigned int i;
	unsigned int j;
	double data;

	file_pt = fopen(filename,"rb");  //rb read binary

	for(i = 0; i < amount; i++)
	{
		for(j = 0; j < amount; j++)
		{
			fread(bytes,sizeof(bytes),1,file_pt); //read 8 bytes, i.e. a double
			data = *((double*)bytes);
			matrix[i * amount + j] = data;
		}
	}

	fclose(file_pt);
}

//read arrays
void read_int_array(const char *filename, int *array, unsigned int amount)
{
	char bytes[4];
	FILE *file_pt;
	unsigned int i;
	int data;

	file_pt = fopen(filename,"rb");  //rb read binary

	for(i = 0; i < amount; i++)
	{
		fread(bytes,sizeof(bytes),1,file_pt); //read 8 bytes, i.e. an int
		data = *((int*)bytes);
		array[i] = data;
	}

	fclose(file_pt);
}

void read_double_array(const char *filename, double *array, unsigned int amount)
{
	char bytes[8];
	FILE *file_pt;
	unsigned int i;
	double data;

	file_pt = fopen(filename,"rb");  //rb read binary

	for(i = 0; i < amount; i++)
	{
		fread(bytes,sizeof(bytes),1,file_pt); //read 8 bytes, i.e. a double
		data = *((double*)bytes);
		array[i] = data;
	}

	fclose(file_pt);
}

//write double array
void write_double_array(const char *filename, double *array, unsigned int amount)
{
	FILE *file_pt;
	unsigned int i;

	file_pt = fopen(filename,"w");  //rb read binary
	fprintf(file_pt, "Current flows: \n");

	for(i = 0; i < amount; i++)
	{
		fprintf(file_pt, "%f \n", array[i]);
	}

	fclose(file_pt);
}

//set an array to a value
// kernel set
__global__ void kernel_set_val(double* A, const double a, int n_el)
{
	// calculate the unique thread index
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	// perform tid-th elements addition
	if (tid < n_el) 
	{
		A[tid] = a;
	}

}

void set_val(double* A, const double a, int n_el)
{
	// declare the number of blocks per grid and the number of threads per block
	int threadsPerBlock,blocksPerGrid;

	// use 1 to 1024 threads per block
	if (n_el < 1024)
	{
		threadsPerBlock = n_el;
		blocksPerGrid   = 1;
	} 
	else 
	{
		threadsPerBlock = 1024;
		blocksPerGrid   = ceil(double(n_el)/double(threadsPerBlock));
	}

	// invoke the kernel
	kernel_set_val<<<blocksPerGrid,threadsPerBlock>>>(A, a, n_el);

}

//scale an array
// kernel scale
__global__ void kernel_scale(double* A, const double a, int n_el)
{
	// calculate the unique thread index
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	// perform tid-th elements addition
	if (tid < n_el) 
	{
		A[tid] *= a;
	}

}

void scale(double* A, const double a, int n_el)
{
	// declare the number of blocks per grid and the number of threads per block
	int threadsPerBlock,blocksPerGrid;

	// use 1 to 1024 threads per block
	if (n_el < 1024)
	{
		threadsPerBlock = n_el;
		blocksPerGrid   = 1;
	} 
	else 
	{
		threadsPerBlock = 1024;
		blocksPerGrid   = ceil(double(n_el)/double(threadsPerBlock));
	}

	// invoke the kernel
	kernel_scale<<<blocksPerGrid,threadsPerBlock>>>(A, a, n_el);

}

//sum of 2, in place
// kernel sum ip
__global__ void kernel_sum_ip(double* A, const double* B, const double a, const double b, int n_el)
{
	// calculate the unique thread index
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	// perform tid-th elements addition
	if (tid < n_el) 
	{
		A[tid] = a * A[tid] + b * B[tid];
	}

}

void scale_sum_ip(double* A, const double* B, const double a, const double b, int n_el)
{
	// declare the number of blocks per grid and the number of threads per block
	int threadsPerBlock,blocksPerGrid;

	// use 1 to 1024 threads per block
	if (n_el < 1024)
	{
		threadsPerBlock = n_el;
		blocksPerGrid   = 1;
	} 
	else 
	{
		threadsPerBlock = 1024;
		blocksPerGrid   = ceil(double(n_el)/double(threadsPerBlock));
	}

	// invoke the kernel
	kernel_sum_ip<<<blocksPerGrid,threadsPerBlock>>>(A, B, a, b, n_el);

}

//sum of 2
// kernel sum
__global__ void kernel_sum(const double* A, const double* B, const double a, const double b, double* C, int n_el)
{
	// calculate the unique thread index
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	// perform tid-th elements addition
	if (tid < n_el) 
	{
		C[tid] = a * A[tid] + b * B[tid];
	}

}

void scale_sum(const double* A, const double* B, const double a, const double b, double* C, int n_el)
{
	// declare the number of blocks per grid and the number of threads per block
	int threadsPerBlock,blocksPerGrid;

	// use 1 to 1024 threads per block
	if (n_el < 1024)
	{
		threadsPerBlock = n_el;
		blocksPerGrid   = 1;
	} 
	else 
	{
		threadsPerBlock = 1024;
		blocksPerGrid   = ceil(double(n_el)/double(threadsPerBlock));
	}

	// invoke the kernel
	kernel_sum<<<blocksPerGrid,threadsPerBlock>>>(A, B, a, b, C, n_el);

}

//sum of 3
// kernel 3 sum
__global__ void kernel_sum_3(const double* A, const double* B, const double* C, const double a, const double b, const double c, double* D, int n_el)
{
	// calculate the unique thread index
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	// perform tid-th elements addition
	if (tid < n_el) 
	{
		D[tid] = a * A[tid] + b * B[tid] + c * C[tid];
	}

}

void scale_sum_3(const double* A, const double* B, const double* C, const double a, const double b, const double c, double* D, int n_el)
{
	// declare the number of blocks per grid and the number of threads per block
	int threadsPerBlock,blocksPerGrid;

	// use 1 to 1024 threads per block
	if (n_el < 1024)
	{
		threadsPerBlock = n_el;
		blocksPerGrid   = 1;
	} 
	else 
	{
		threadsPerBlock = 1024;
		blocksPerGrid   = ceil(double(n_el)/double(threadsPerBlock));
	}

	// invoke the kernel
	kernel_sum_3<<<blocksPerGrid,threadsPerBlock>>>(A, B, C, a, b, c, D, n_el);

}

//expansion sparse array into full array
__global__ void kernel_array_expansion(const int* col_ind, const double* conductance, const int start_position, double* target_array, int n_el)
{
	// calculate the unique thread index
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	int index = start_position + tid;
	int location = col_ind[index];

	// perform tid-th elements addition
	if (tid < n_el) 
	{
		target_array[location] = conductance[index];
	}
}

void array_expansion(const int* col_ind, const double* conductance, const int row, const int* spacing, double* target_array, int n_el)
{
	// declare the number of blocks per grid and the number of threads per block
	int threadsPerBlock,blocksPerGrid;

	int start = spacing[row];
	int end = spacing[row + 1];

	int n_el_exp = end - start;
	
	// use 1 to 1024 threads per block
	if (n_el_exp < 1024)
	{
		threadsPerBlock = n_el_exp;
		blocksPerGrid   = 1;
	} 
	else 
	{
		threadsPerBlock = 1024;
		blocksPerGrid   = ceil(double(n_el_exp)/double(threadsPerBlock));
	}

	kernel_array_expansion<<<blocksPerGrid,threadsPerBlock>>>(col_ind, conductance, start, target_array, n_el_exp);
}

//X * cond diff calculation
//kernel X * cond diff calculation
__global__ void kernel_calculate_X_cond_diff(const int* row_ind, const int* col_ind, const double* conductance, const double* potentials, double* X_cond_diff, int n_el)
{
	// calculate the unique thread index
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	
	// perform tid-th elements addition
	if (tid < n_el) 
	{
		int row = row_ind[tid];
		int col = col_ind[tid];

		X_cond_diff[tid] = abs((potentials[row] - potentials[col]) * conductance[tid]) / 2;
	}

}

void calculate_X_cond_diff(const int* row_ind, const int* col_ind, const double* conductance, const double* potentials, double* X_cond_diff, int n_el)
{
	// declare the number of blocks per grid and the number of threads per block
	int threadsPerBlock,blocksPerGrid;

	// use 1 to 1024 threads per block
	if (n_el < 1024)
	{
		threadsPerBlock = n_el;
		blocksPerGrid   = 1;
	} 
	else 
	{
		threadsPerBlock = 1024;
		blocksPerGrid   = ceil(double(n_el)/double(threadsPerBlock));
	}

	// invoke the kernel
	kernel_calculate_X_cond_diff<<<blocksPerGrid,threadsPerBlock>>>(row_ind, col_ind, conductance, potentials, X_cond_diff, n_el);
}

//current calculate
// kernel calculate current
__global__ void kernel_calculate_current(const int* LUT, const int LUT_start, const int offset, double* X_cond_diff, int n_el)
{
	// calculate the unique thread index
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	int operation_location = LUT[LUT_start + tid];

	if (tid < n_el) 
	{
		X_cond_diff[operation_location] += X_cond_diff[operation_location + offset];
		__syncthreads();
	}
}

//kernel copy current
__global__ void kernel_copy_current(const int* spacing, double* X_cond_diff, double* current, int n_el)
{
	// calculate the unique thread index
	int tid = blockDim.x * blockIdx.x + threadIdx.x;

	if (tid < n_el) 
	{
		int copy_location = spacing[tid];
		current[tid] = X_cond_diff[copy_location];
	}
}

void calculate_current(const int* spacing, const int* LUT, const int* LUT_positions, double* X_cond_diff, double* current, int n_el_copy)
{
	// declare the number of blocks per grid and the number of threads per block
	int threadsPerBlock,blocksPerGrid;
	int i;
	int sum_offset = 1;
	int LUT_start;
	int LUT_end;
	int n_el_calc;

	// invoke the kernel
	for(i = 0; i < number_of_rounds; i++)
	{
		//calculate the total number of threads needed
		LUT_start = LUT_positions[i];
		LUT_end = LUT_positions[i + 1];
		n_el_calc = LUT_end - LUT_start;

		//calculate
		// use 1 to 1024 threads per block
		if (n_el_calc < 1024)
		{
			threadsPerBlock = n_el_calc;
			blocksPerGrid   = 1;
		} 
		else 
		{
			threadsPerBlock = 1024;
			blocksPerGrid   = ceil(double(n_el_calc)/double(threadsPerBlock));
		}

		kernel_calculate_current<<<blocksPerGrid,threadsPerBlock>>>(LUT, LUT_start, sum_offset, X_cond_diff, n_el_calc);

		sum_offset *= 2;
	}

	//copy
	// use 1 to 1024 threads per block
	if (n_el_copy < 1024)
	{
		threadsPerBlock = n_el_copy;
		blocksPerGrid   = 1;
	} 
	else 
	{
		threadsPerBlock = 1024;
		blocksPerGrid   = ceil(double(n_el_copy)/double(threadsPerBlock));
	}

	kernel_copy_current<<<blocksPerGrid,threadsPerBlock>>>(spacing, X_cond_diff, current, n_el_copy);
}

int main()
{
	clock_t start, finish;
	clock_t Au_start, Au_finish;
	clock_t current_start, current_finish;
	double time_cost;

	cublasHandle_t handle;  
    	cublasStatus_t status = CUBLAS_STATUS_SUCCESS;  

	status = cublasCreate(&handle);

	int i;
	int j;

	int load_ind;

	double p;

	//temporary constants created for sum calc
	double const_1;
	double const_2;
	double const_3;

	double *A_inv;
	double *g_A_inv_part;
	double *A_diag;

	A_inv = (double *)malloc(m_size*m_size*sizeof(double));
	cudaMalloc((void **)&g_A_inv_part,col_load_max*m_size*sizeof(double));
	A_diag = (double *)malloc(m_size*sizeof(double));

	double *g_A_i;
	double *g_A_inv_1;
	double *g_A_inv_i;

	cudaMalloc((void **)&g_A_i,m_size*sizeof(double));
	cudaMalloc((void **)&g_A_inv_1,m_size*sizeof(double));
	cudaMalloc((void **)&g_A_inv_i,m_size*sizeof(double));

	double *g_A_inv_j;

	cudaMalloc((void **)&g_A_inv_j,m_size*sizeof(double));

	double *ep;
	double *g_ep;
	double *g_temp_1;
	double *g_temp_2;
	double *g_Au_2;

	ep = (double *)malloc(m_size*sizeof(double));
	cudaMalloc((void **)&g_ep,m_size*sizeof(double));
	cudaMalloc((void **)&g_temp_1,m_size*sizeof(double));
	cudaMalloc((void **)&g_temp_2,m_size*sizeof(double));
	cudaMalloc((void **)&g_Au_2,m_size*sizeof(double));

	double *g_A_at_at_inv_v_2;
	double *g_A_at_at_inv_u_2;

	cudaMalloc((void **)&g_A_at_at_inv_v_2,m_size*sizeof(double));
	cudaMalloc((void **)&g_A_at_at_inv_u_2,m_size*sizeof(double));

	double *g_sigma;
	double *g_X;
	double *g_X_cond_diff;

	cudaMalloc((void **)&g_sigma,m_size*sizeof(double));
	cudaMalloc((void **)&g_X,m_size*sizeof(double));
	cudaMalloc((void **)&g_X_cond_diff,element_num*sizeof(double));

	int *spacing;
	int *g_spacing;
	int *row_ind;
	int *g_row_ind;
	int *col_ind;
	int *g_col_ind;
	int *LUT;
	int *g_LUT;
	int *LUT_positions;
	double *conductance;
	double *g_conductance;
	double *g_current;
	double *total_current;
	double *g_total_current;

	spacing = (int *)malloc((m_size+1)*sizeof(int));
	cudaMalloc((void **)&g_spacing,(m_size+1)*sizeof(int));
	row_ind = (int *)malloc(element_num*sizeof(int));
	cudaMalloc((void **)&g_row_ind,element_num*sizeof(int));
	col_ind = (int *)malloc(element_num*sizeof(int));
	cudaMalloc((void **)&g_col_ind,element_num*sizeof(int));
	LUT = (int *)malloc(LUT_size*sizeof(int));
	cudaMalloc((void **)&g_LUT,LUT_size*sizeof(int));
	LUT_positions = (int *)malloc((number_of_rounds+1)*sizeof(int));
	conductance = (double *)malloc(element_num*sizeof(double));
	cudaMalloc((void **)&g_conductance,element_num*sizeof(double));
	cudaMalloc((void **)&g_current,m_size*sizeof(double));
	total_current = (double *)malloc(m_size*sizeof(double));
	cudaMalloc((void **)&g_total_current,m_size*sizeof(double));

	double namda_1;
	double mu_1;
	double A_inv_ii_1;
	double v_A_inv_star_u_1;

	double namda_2;
	double mu_2;
	double A_inv_ii_2;
	double v_A_inv_star_u_2;

	double a_1;
	double b_1;
	double c_1;
	double d_1;

	double q_1;
    	double q_2;
    	double q_3;
    	double q_4;

	double a_2;
    	double b_2;
    	double c_2;
    	double d_2;

	//loading the matrices and arrays
	read_double_array("chrom_1_initial_matrix_diag.bin", A_diag, m_size);
	read_matrix("chrom_1_initial_inverse_matrix.bin", A_inv, m_size);
	printf("Initial inverse matrix reading complete\n");

	read_int_array("chrom_1_occupied_spacings.bin", spacing, m_size+1);
	read_int_array("chrom_1_occupied_row_indices.bin", row_ind, element_num);
	read_int_array("chrom_1_occupied_col_indices.bin", col_ind, element_num);
	read_int_array("chrom_1_LUT.bin", LUT, LUT_size);
	read_int_array("chrom_1_LUT_positions.bin", LUT_positions, number_of_rounds+1);
	read_double_array("chrom_1_occupied_values.bin", conductance, element_num);
	printf("Occupied matrices reading complete\n");
	printf("\n");

	//calculate p = - A(1,1) / 4
	p = - A_diag[0] / 4;

	//copy in or make the things that always needed
	cudaMemcpy(g_A_inv_1, A_inv, m_size*sizeof(double), cudaMemcpyHostToDevice);

	cudaMemcpy(g_spacing, spacing, (m_size+1)*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(g_row_ind, row_ind, element_num*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(g_col_ind, col_ind, element_num*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(g_LUT, LUT, LUT_size*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(g_conductance, conductance, element_num*sizeof(double), cudaMemcpyHostToDevice);

	printf("Finished major loadings\n");

	set_val(g_total_current, 0, m_size);

	//loop through rows
	for(i = 0; i < m_size; i++)
	{
		start = clock();

		//copy in or make i dependent items
		//construct g_A_i
		set_val(g_A_i, 0, m_size);
		array_expansion(g_col_ind, g_conductance, i, spacing, g_A_i, m_size);

		cudaMemcpy(g_A_inv_i, A_inv+i*m_size, m_size*sizeof(double), cudaMemcpyHostToDevice);

		printf("Finished i related loadings\n");

		//create temp vals for calculation
		double A_inv_1_1 = A_inv[0];
		double A_inv_1_i = A_inv[i];
		double A_inv_i_i = A_inv[i*m_size + i];

		namda_1 = p * A_inv_1_1 + A_inv_1_i;
		mu_1 = p * p * A_inv_1_1 + 2 * p * A_inv_1_i + A_inv_i_i;

		A_inv_ii_1 = A_inv_1_1;

		v_A_inv_star_u_1 = namda_1 - mu_1 * A_inv_ii_1 / (1 + namda_1);

		a_1 = - 1 / (1 + v_A_inv_star_u_1);
    		b_1 = - 1 / (1 + namda_1) - (mu_1 * A_inv_ii_1 / ((1 + namda_1) * (1 + namda_1))) / (1 + v_A_inv_star_u_1);
    		c_1 = (A_inv_ii_1 / (1 + namda_1)) / (1 + v_A_inv_star_u_1);
    		d_1 = (mu_1 / (1 + namda_1)) / (1 + v_A_inv_star_u_1);

		//make ep
		cudaMemcpy(ep, g_A_i, m_size*sizeof(double), cudaMemcpyDeviceToHost);
		ep[0] += 1;
		ep[i] = 0.5 * (A_diag[i] - 1);
		cudaMemcpy(g_ep, ep, m_size*sizeof(double), cudaMemcpyHostToDevice);

		//ep = -ep
		scale(g_ep, -1, m_size);

		printf("Finish creating ep\n");

		q_1 = a_1 * p + c_1 * p * p + b_1 * p + d_1;
    		q_2 = c_1 * p + b_1;
    		q_3 = a_1 + c_1 * p;
    		q_4 = c_1;

		//temp_1 = q_1 * A_inv(1,:) + q_2 * A_inv(i,:);
		const_1 = q_1;
		const_2 = q_2;
		scale_sum(g_A_inv_1, g_A_inv_i, const_1, const_2, g_temp_1, m_size);

		//temp_2 = q_3 * A_inv(1,:) + q_4 * A_inv(i,:);
		const_1 = q_3;
		const_2 = q_4;
		scale_sum(g_A_inv_1, g_A_inv_i, const_1, const_2, g_temp_2, m_size);

		Au_start = clock();

		//zero it
		set_val(g_Au_2, 0, m_size);

		double alpha = 1;
		double beta = 0;

		//Au_2 = (A_inv * ep')';
		//in parts
		for(load_ind = 0; load_ind < number_of_loads; load_ind++)
		{
			//number of columns to be loaded
			int num_col_load;

			//before reaching the end
			if(load_ind != (number_of_loads - 1))
			{
				num_col_load = col_load_max;
			}
			//reaching the end
			else
			{
				num_col_load = m_size - load_ind * col_load_max;
			}

			//memcpy the part from disk
			cudaMemcpy(g_A_inv_part, A_inv+load_ind*col_load_max*m_size, num_col_load*m_size*sizeof(double), cudaMemcpyHostToDevice);

			//calculate part of the matrix-vector product
			status = cublasDgemv(handle, CUBLAS_OP_T, m_size, num_col_load, &alpha, g_A_inv_part, m_size, g_ep, 1, &beta, g_Au_2+load_ind*col_load_max, 1);
			
		}

		Au_finish = clock();
		time_cost = (double)(Au_finish - Au_start) / CLOCKS_PER_SEC;
		printf("Au matrix took %f seconds to execute\n", time_cost);

		//A_at_at_inv_v_2 = A_inv(i,:) + temp_1 * A_inv(1,i) + temp_2 * A_inv(i,i);
		const_1 = 1;
		const_2 = A_inv_1_i;
		const_3 = A_inv_i_i;
		scale_sum_3(g_A_inv_i, g_temp_1, g_temp_2, const_1, const_2, const_3, g_A_at_at_inv_v_2, m_size);

		//create temp vals for calculation
		double Au_2_1;
		double Au_2_i;

		cudaMemcpy(&Au_2_1, g_Au_2, sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(&Au_2_i, g_Au_2 + i, sizeof(double), cudaMemcpyDeviceToHost);

		//A_at_at_inv_u_2 = Au_2 + temp_1 * Au_2(1) + temp_2 * Au_2(i);
		const_1 = 1;
		const_2 = Au_2_1;
		const_3 = Au_2_i;
		scale_sum_3(g_Au_2, g_temp_1, g_temp_2, const_1, const_2, const_3, g_A_at_at_inv_u_2, m_size);

		//namda_2 = ep * A_at_at_inv_v_2';
    		//mu_2 = ep * A_at_at_inv_u_2';
		status = cublasDdot(handle, m_size, g_ep, 1, g_A_at_at_inv_v_2, 1, &namda_2);
		status = cublasDdot(handle, m_size, g_ep, 1, g_A_at_at_inv_u_2, 1, &mu_2);

		//create temp vals for calculation
		double A_at_at_inv_v_2_i;

		cudaMemcpy(&A_at_at_inv_v_2_i, g_A_at_at_inv_v_2 + i, sizeof(double), cudaMemcpyDeviceToHost);

		A_inv_ii_2 = A_at_at_inv_v_2_i;
		v_A_inv_star_u_2 = namda_2 - mu_2 * A_inv_ii_2 / (1 + namda_2);

		a_2 = - 1 / (1 + v_A_inv_star_u_2);
		b_2 = - 1 / (1 + namda_2) - (mu_2 * A_inv_ii_2 / ((1 + namda_2) * (1 + namda_2))) / (1 + v_A_inv_star_u_2);
		c_2 = (A_inv_ii_2 / (1 + namda_2)) / (1 + v_A_inv_star_u_2);
		d_2 = (mu_2 / (1 + namda_2)) / (1 + v_A_inv_star_u_2);

		printf("v_A_inv_star_u_2: %f \n", v_A_inv_star_u_2);
		printf("namda_2: %f \n", namda_2);

		printf("a_2: %f \n", a_2);
		printf("b_2: %f \n", b_2);
		printf("c_2: %f \n", c_2);
		printf("d_2: %f \n", d_2);

		for(j = i+1; j < m_size; j++)
		{
			current_start = clock();

			set_val(g_current, 0, m_size);

			//create temp vals for calculation
			double A_at_at_inv_v_2_j;
			double A_at_at_inv_u_2_j;

			cudaMemcpy(&A_at_at_inv_v_2_j, g_A_at_at_inv_v_2 + j, sizeof(double), cudaMemcpyDeviceToHost);
			cudaMemcpy(&A_at_at_inv_u_2_j, g_A_at_at_inv_u_2 + j, sizeof(double), cudaMemcpyDeviceToHost);

			//acquire A_inv[j]
			cudaMemcpy(g_A_inv_j, A_inv+j*m_size, m_size*sizeof(double), cudaMemcpyHostToDevice);

			//get sigma: sigma = A_inv(j,:) 
			//+ (a_2 * A_at_at_inv_v_2(j) + c_2 * A_at_at_inv_u_2(j)) * Au_2  
			//+ (b_2 * A_at_at_inv_u_2(j) + d_2 * A_at_at_inv_v_2(j)) * A_inv(i,:);
			const_1 = 1;
			const_2 = a_2 * A_at_at_inv_v_2_j + c_2 * A_at_at_inv_u_2_j;
			const_3 = b_2 * A_at_at_inv_u_2_j + d_2 * A_at_at_inv_v_2_j;

			scale_sum_3(g_A_inv_j, g_Au_2, g_A_inv_i, const_1, const_2, const_3, g_sigma, m_size);			

			//create temp vals for calculation
			double sigma_1;
			double sigma_i;

			cudaMemcpy(&sigma_1, g_sigma, sizeof(double), cudaMemcpyDeviceToHost);
			cudaMemcpy(&sigma_i, g_sigma + i, sizeof(double), cudaMemcpyDeviceToHost);

			//X = sigma + sigma(1) * temp_1 + sigma(i) * temp_2;
			const_1 = 1;
			const_2 = sigma_1;
			const_3 = sigma_i;

			scale_sum_3(g_sigma, g_temp_1, g_temp_2, const_1, const_2, const_3, g_X, m_size);	

			//calculate single current
			calculate_X_cond_diff(g_row_ind, g_col_ind, g_conductance, g_X, g_X_cond_diff, element_num);
			calculate_current(g_spacing, g_LUT, LUT_positions, g_X_cond_diff, g_current, m_size);

			//i_nodes([j i]) = 0;
			cudaMemset(g_current + i, 0, sizeof(double));
			cudaMemset(g_current + j, 0, sizeof(double));

			scale_sum_ip(g_total_current, g_current, 1, 1, m_size);

			current_finish = clock();
			time_cost = (double)(current_finish - current_start) / CLOCKS_PER_SEC;
			//printf("Current flow took %f seconds to execute\n", time_cost);
		}

		finish = clock();

		time_cost = (double)(finish - start) / CLOCKS_PER_SEC;
		printf("Loop %d took %f seconds to execute\n", i, time_cost);

		printf("\n");
	}

	cudaMemcpy(total_current, g_total_current, m_size*sizeof(double), cudaMemcpyDeviceToHost);

	write_double_array("current_flow", total_current, m_size);

	free(A_diag);
	free(A_inv);
	cudaFree(g_A_inv_part);

	cudaFree(g_A_i);
	cudaFree(g_A_inv_1);
	cudaFree(g_A_inv_i);

	cudaFree(g_A_inv_j);

	free(ep);
	cudaFree(g_ep);
	cudaFree(g_temp_1);
	cudaFree(g_temp_2);
	cudaFree(g_Au_2);

	cudaFree(g_A_at_at_inv_v_2);
	cudaFree(g_A_at_at_inv_u_2);

	cudaFree(g_sigma);
	cudaFree(g_X);
	cudaFree(g_X_cond_diff);

	free(spacing);
	cudaFree(g_spacing);
	free(row_ind);
	cudaFree(g_row_ind);
	free(col_ind);
	cudaFree(g_col_ind);
	free(LUT);
	cudaFree(g_LUT);
	free(LUT_positions);
	free(conductance);
	cudaFree(g_conductance);
	cudaFree(g_current);
	free(total_current);
	cudaFree(g_total_current);

	cublasDestroy(handle);

}
