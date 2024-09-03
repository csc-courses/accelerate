#include<cuda_runtime.h>
#include<stdio.h>
#include<math.h>

/*
 * sum all elements in a vector
 * author: bjr
 * last update:
 *		2 sep 2022 0bjr: initial
 */


#ifndef N_ELEM
#define N_ELEM 32
#endif

// cuda kernel
__global__ void fold_array(float * a, int d) {
	int i = threadIdx.x + blockIdx.x * blockDim.x ;
	a[i] += a[i+d] ;
	return ;
}

// host routines

void initialData(float *ip, int size) {
	time_t t ;
	int i ;
	static int j = 0 ;

	if (!j++) srand ((unsigned)time(&t)) ;
	for (i=0; i<size; i++) {
		ip[i] = (float) ( rand() & 0xFF ) / 10.0f ;
	}
	return ;
}

#define PRINT_I 6
#define PRINT_L 2

void printData(const char * s, float *ip, int n) {
	int i, k ;
	int f = PRINT_I ;
	int l = PRINT_L ;
	printf("%s\t",s) ;
	if (n<=f) {
		for (i=0;i<n;i++) {
			printf("%5.2f\t", ip[i]) ;
		}
		printf("\n") ;
		return ;
	}
	for (i=0;i<f;i++) {
		printf("%5.2f\t", ip[i]) ;
	}
	printf("\t...\t") ;
	k = n - l ;
	if (k<f) k = f ;
	for (i=k;i<n;i++) {
		printf("%5.2f\t", ip[i]) ;
	}
	printf("\n") ;
	return ;
}

float distance(float * a, float * b, float *c, int n) {
	float f, dist = 0.0 ;
	int i ;
	for (i=0;i<n;i++) {
		f = b[i] * c[i] - a[i] ;
		dist += f*f ;
	}
	return sqrt(dist) ;
}

int log_2_round(int n) {
	int d = 1 ;
	n >>= 1 ;
	while (n){
		n >>= 1 ;
		d <<= 1 ;
	}
	return d  ;
}

int main(int argc, char * argv[]) {
	int dev = 0 ;
	int n = N_ELEM ;
	int n_bytes = n * sizeof(float) ;
	float * h_a ;
	float * d_a ;

	cudaSetDevice(dev) ;

	h_a = (float *) malloc(n_bytes) ;
	cudaMalloc((float **)&d_a, n_bytes) ;

	initialData(h_a, n ) ;

	cudaMemcpy(d_a, h_a n_bytes, cudaMemcpyHostToDevice) ;

	n_1 = log_2_round(n) ;
	if (n>n-1) {
		fold_array<<<(1,(n-n1))>>>(d_a,n_1) ;
	}
	d >>=1 ;
	while (d) {
		fold_array<<<(1,d)>>>(d_a,d) ;
		d >>= 1;
	}
	
	cudaMemcpy(h_a, d_a, n_bytes, cudaMemcpyDeviceToHost) ;

	printf("n = %d\n", n) ;
	printData("a =\n ", h_b,n) ;

	cudaFree(d_a) ;
	free(h_a) ;

	return 0 ;
}

