#include<cuda_runtime.h>
#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include<string.h>
#include<assert.h>


/*
 * prefix sum
 * author: bjr
 * date: 28 feb 2019
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

#define USAGE_MESSAGE "cat _file_ | prefix-sum _blocks_ _threads_"

// DEVICE

__shared__ int is_verbose ;

__global__ void first_stage(int * a, int b, int n) {
	//
	// create a tree and sum spans upwards
	//
	int thread = threadIdx.x ;
	int block = blockIdx.x ;
	int offset = block * blockDim.x ;
	int level ;
	int loc = offset + thread ;

	is_verbose = IS_VERBOSE ;
	if (is_verbose) {
		printf("thread %d, block %d\n", thread, block) ; 
	}

	level = 1 ;
	while  (level<n) {
		if (level&thread) break ;
		a[loc] += a[loc+level] ;
		level *= 2 ;
		__syncthreads() ;
	}
	return ;
}

__global__ void second_stage(int * a, int b, int n) {
	//
	// return back down the tree completing the partial sums
	//
	int thread = threadIdx.x ;
	int block = blockIdx.x ;
	int offset = block * blockDim.x ;
	int level ;
	int loc = offset + thread ;

	level = n/4 ; // start with these
	while (level>0) {
		if ( thread%level==0 ) 
			if ( thread/level & 1 ) 
				if (thread+level < blockDim.x ) 
					a[loc] += a[loc+level] ; 
		level /= 2 ;
		__syncthreads() ;
	}
	return ;
}

__global__ void third_stage(int * a, int b, int t) {
	//
	// same thing as first stage, but over the 0th element of each block
	//
	int thread = threadIdx.x ;
	int block = blockIdx.x ;
	int offset = block * blockDim.x ;
	int level ;
	int loc = offset + thread ;
	
	level = 1 ;

	while  (level<t) {
		if (level&thread) break ;
		a[thread*b] += a[(thread+level)*b] ;
		level *= 2 ;
		__syncthreads() ;
	}
	return ;
}

__global__ void fourth_stage(int * a, int b, int t) {
	//
	// same thing as second stage, but over the 0th element of each block
	//
	int thread = threadIdx.x ;
	int block = blockIdx.x ;
	int offset = block * blockDim.x ;
	int level ;
	int loc = offset + thread ;

	level = t/4 ; // start with these
	while (level>0) {
		if ( thread%level==0 ) 
			if ( thread/level & 1 ) 
				if (thread+level < t ) 
					a[thread*b] += a[(thread+level)*b] ; 
		level /= 2 ;
		__syncthreads() ;
	}
	return ;
}

__global__ void fifth_stage(int * a, int b, int t) {
	//
	// combine the block-wise some in the 0th element of the blocks
	// with the prefix sums summed up inside a block
	//
	int thread = threadIdx.x ;
	int block = blockIdx.x ;
	int offset = block * blockDim.x ;
	int level ;
	int loc = offset + thread ;

	if (block==(b-1)) return ;
	if (loc%t) {
		a[loc] += a[(block+1)*t] ;
	}
 	return ;
}


// HOST

void print_numbers(const char * s, int * a_i, int n) {
	int i ; 
	printf("%s (%d):\t",s, n ) ;
	for (i=0;i<n;i++) printf("%d ", a_i[i]) ;
	printf("\n") ;
	return;
}

void prefix_sum(int *a, int n) {
	int i ;
	for (i=n-2;i>=0;i--) {
		a[i] += a[i+1] ;
	}
	return ;
}

int array_eq( int *a, int *b, int n) {
	int i ;
	for (i=0;i<n;i++) if (a[i]!=b[i]) return FALSE ;
	return TRUE ;
}

int power_of_two(int x) {
	while (! (x&1) ) x /= 2 ;
	return x==1 ;
}

int main(int argc, char * argv[]) {

	char buffer[BUFFER_N] ;
	int buf_n = BUFFER_N ;
	int ch ;
	int i ;
	int is_verbose = 0 ;

	int * h_a ;
	int * h_b ;
	int n_threads ;
	int n_blocks ;
	int n_num ;
	int n_bytes ;

	int dev = 0 ;
	int * d_a ;

	cudaSetDevice(dev) ;

	is_verbose = IS_VERBOSE ;

	while ((ch = getopt(argc, argv, "v")) != -1) {
		switch(ch) {
		case 'v':
			is_verbose = 1 ;
			break ;
		case '?':
		default:
			printf(USAGE_MESSAGE) ;
			return 0 ;
		}
	}
	argc -= optind;
	argv += optind;

	if ( argc!= 2 ) {
		fprintf(stderr,"%s\n",USAGE_MESSAGE) ;
		exit(0) ;
	}
	n_blocks = atoi(argv[0]) ;
	n_threads = atoi(argv[1]) ;
	assert(power_of_two(n_blocks));
	assert(power_of_two(n_threads)); 

	if ( is_verbose ) {
		printf("blocks: %d, threads: %d\n", n_blocks, n_threads) ;
	}

	n_num = n_blocks * n_threads ;
	n_bytes = n_num * sizeof(int) ;

	h_a = (int *) malloc(n_bytes) ;
	assert(h_a) ;
	h_b = (int *) malloc(n_bytes) ;
	assert(h_b) ;

	i = 0 ;
	while( fgets(buffer, buf_n, stdin)) {
		char * s ;
		s = strtok(buffer, SEP_CHAR) ;
		while (s) {
			assert(i<n_num) ;
			h_a[i++]= atoi(s) ;
			s = strtok(NULL, SEP_CHAR) ;
		}
	}
	assert( i == n_num ) ;
	print_numbers("In", h_a, n_num) ;

	cudaMalloc((int **)&d_a, n_bytes) ;
	cudaMemcpy(d_a, h_a, n_bytes, cudaMemcpyHostToDevice) ;
	
	// create prefix sum per block
	first_stage <<<n_blocks,n_threads>>>(d_a, n_blocks, n_threads) ;
	second_stage <<<n_blocks,n_threads>>>(d_a, n_blocks, n_threads) ;

	// create prefix sum over the "summary" of the block, i.e. the sum of 
	// all elements in the block
	
	// N.B. these somewhat hackish arrangements create a block of threads as numerous
	// as there are overall bloicks
	third_stage <<<1,n_blocks>>>(d_a, n_threads, n_blocks) ;
	fourth_stage <<<1,n_blocks>>>(d_a, n_threads, n_blocks) ;
	
	// combine the prefix sums inside the block with the prefix sums over the 
	// block summaries
	fifth_stage <<<n_blocks,n_threads>>>(d_a, n_blocks, n_threads) ;
	

	cudaMemcpy(h_b, d_a, n_bytes, cudaMemcpyDeviceToHost) ;
	print_numbers("Out", h_b, n_num) ;

	prefix_sum(h_a, n_num) ;
	assert( array_eq(h_a, h_b, n_num) == TRUE ) ;
	printf("\n") ;

	cudaFree(d_a) ;

	free(h_a) ;
	free(h_b) ;

	return 0 ;
}

