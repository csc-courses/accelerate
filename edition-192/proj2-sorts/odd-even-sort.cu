#include<cuda_runtime.h>
#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include<string.h>
#include<assert.h>


/*
 * even odd sort
 * author: bjr
 * date: 6 feb 2019
 *
 */


#define BUFFER_N 1024
#define SEP_CHAR " \t\n,"
#define MAX_NUMBERS 1024

#ifndef IS_VERBOSE
#define IS_VERBOSE 0
#endif

#define TRUE 1
#define FALSE 0

#define USAGE_MESSAGE "cat _file_ | odd-even-sort"

__shared__ int is_verbose ;

__device__ void swap(int * a, int i) {
	int t ;

	if (a[i]>a[i+1]) {
		t = a[i] ;
		a[i] = a[i+1] ;
		a[i+1] = t ;
	}
}

__global__ void transposition_stage(int * a, int is_even, int n) {
	int thread = threadIdx.x ;

	is_verbose = IS_VERBOSE ;
	if (is_verbose) {
		printf("thread %d, a[%d] = %d\n", thread, thread, a[thread]) ; 
	}

	if (thread+1<n) {
		if (is_even == !(thread%2)) swap(a,thread) ;
	}	
	return ;
}

// *****************************************
// HOST
// *****************************************


int test_array(int * a, int k) {
	int i ;
	for (i=1;i<k;i++) {
		if (a[i-1]>a[i]) return FALSE ;
	}
	return TRUE ;
}

void print_numbers(const char * s, int * a_i, int n) {
	int i ; 
	printf("%s:\t",s ) ;
	for (i=0;i<n;i++) printf("%d ", a_i[i]) ;
	printf(" (%d numbers)\n",n) ;
	return;
}

int main(int argc, char * argv[]) {

	char buffer[BUFFER_N] ;
	int buf_n = BUFFER_N ;
	int numbers[MAX_NUMBERS] ;
	int n_num ;
	int n_bytes ;
	int stage ;

	int dev = 0 ;
	int * d_a ;

	cudaSetDevice(dev) ;

	n_num = 0 ;
	while( fgets(buffer, buf_n, stdin)) {
		char * s ;
		s = strtok(buffer, SEP_CHAR) ;
		while (s) {
			numbers[n_num++]= atoi(s) ;
			s = strtok(NULL, SEP_CHAR) ;
			assert(n_num<MAX_NUMBERS) ;
		}
	}
	print_numbers("In", numbers, n_num) ;

	n_bytes = n_num * sizeof(int) ;
	cudaMalloc((int **)&d_a, n_bytes) ;
	cudaMemcpy(d_a, numbers, n_bytes, cudaMemcpyHostToDevice) ;
	
	for (stage=0;stage<n_num;stage++) {
		transposition_stage<<<1,n_num>>>(d_a, (stage%2==0), n_num) ;
	}

	cudaMemcpy(numbers, d_a, n_bytes, cudaMemcpyDeviceToHost) ;
	print_numbers("Out", numbers, n_num) ;
	printf("\n") ;
	assert(test_array(numbers, n_num)==TRUE) ;

	cudaFree(d_a) ;

	return 0 ;
}

