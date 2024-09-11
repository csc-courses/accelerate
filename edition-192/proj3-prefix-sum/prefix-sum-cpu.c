#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include<string.h>
#include<assert.h>


/*
 * prefix sum CPU
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
	int n_threads ;
	int n_blocks ;
	int n_num ;
	int n_bytes ;

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

	if (is_verbose) print_numbers("In", h_a, n_num) ;
	prefix_sum(h_a, n_num) ;
	if (is_verbose) print_numbers("Out", h_a, n_num) ;

	free(h_a) ;
	return 0 ;
}

