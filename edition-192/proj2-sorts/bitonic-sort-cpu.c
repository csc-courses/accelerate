#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include<string.h>
#include<assert.h>

#define DOWN 1
#define UP -1
#define FALSE 0
#define TRUE 1

/*
 * bitonic sorting network, iterative
 * author: bjr
 * date: 30 oct 3018
 *
 * This code implements the biotonic sorting.
 * Input will be round up to a power of 2.
 *
 * The diagram shows 3 types of blocks in the sorting network:
 *   blue, green and red.
 * In phase1, various levels begin with two numbers at time, then
 * 4, then 8. The blocks alternative blue, green, blue, green. 
 * The last level has just a blue block.
 *
 * A red block is the fine structure inside either a blue or green
 * block. They are recursive and all go the same direction (no alternation).
 * If this direction is DOWN, they are implementing a blue block, if
 * UP they are implementing a gree block.
 *
 */


static int is_2power(int n) {
	while (!(n&1)) {
		n/=2 ;
	}
	return n==1 ;
}


// bitonic sorting

void swap(int *i1, int *i2, int d ){
	if (d==UP) {
		if (*i1<*i2) {	
			int t ;
			t = *i1 ;
			*i1 = *i2 ;
			*i2 = t ;
		}
	} else { // DOWN
		if (*i1>*i2) {
			int t ;
			t = *i1 ;
			*i1 = *i2 ;
			*i2 = t ;
		}
	}
	return ;
}

void sort_red_block(int * a, int n, int direction) {
	int i ;

	if (n<2) return ;
	for (i=0;i<n/2;i++) {
		swap(a+i,a+i+n/2,direction) ;
	}
	sort_red_block(a, n/2, direction) ;
	sort_red_block(a+n/2, n/2, direction ) ;
	return ;
}

void sort_blue_block(int * a, int n) {
	sort_red_block(a,n,DOWN) ;
	return ;
}

void sort_green_block(int * a,int n) {
	sort_red_block(a,n,UP) ;
	return ;
}



void sort_bitonic(int *a, int n) {
	int level = 2 ;
	int loc ;
	assert(is_2power(n)) ;
	while (level<n) {
		loc = 0 ;
		while (loc<n) {
			sort_blue_block(a+loc,level) ;
			sort_green_block(a+loc+level,level) ;
			loc += 2*level ;
		}
		level *= 2 ;
	}	
	assert(level==n) ;
	sort_blue_block(a,n) ;
	return ;
}





// test and main code


#define BUFFER_N 1024
#define SEP_CHAR " \t\n,"
#define MAX_NUMBERS 1024

#define USAGE_MESSAGE "bitonic-sort [-vF] [_01-test-size_]"

int is_verbose = 0 ;

void print_numbers(char * s, int * a_i, int n) {
	int i ; 
	printf("%s:\t",s ) ;
	for (i=0;i<n;i++) printf("%d ", a_i[i]) ;
	printf(" (%d numbers)\n",n) ;
	return;
}

int ipow(int k) { 
	return 1<<k ;
}

int pad_to_2power(int * a, int n) {
	int p2 = 1 ;
	int j ;
	while (p2<n) {
		p2 *= 2 ;
	}
	if (p2==n) return n ;
	assert(p2<=MAX_NUMBERS) ;
	for (j=n;j<p2;j++) {
		a[j] = 0 ;
	}
	return p2 ;
}

int * zero_one_gen(int * a, int k) {
	// on first call, a is non-null and k is the
	// length of a.
	// zero fills a and returns
	// on subsequent calls, gives next 0-1 vector
	// in sequence.
	// returns NULL when sequence is exhausted

	static int * a_s = NULL ;
	static int k_s ;
	static int g ;
	int i, j ;

	if (!a_s) {
		if (!a) return NULL ;
		a_s = a ;
		k_s = k ;
		g = 0 ;
	}

	if (g==ipow(k_s)) return NULL ;

	j = g ;
	for (i=0;i<k_s;i++) {
		a_s[i] = j%2 ;
		j /= 2 ;
	}

	g++ ;
	return a_s ;
}

int test_array(int * a, int k) {
	int i ;
	for (i=1;i<k;i++) {
		if (a[i-1]>a[i]) return FALSE ;
	}
	return TRUE ;
}

int zero_one_test(int k) {
	int n = ipow(k) ;
	int * a = (int *) malloc(n*sizeof(int)) ;

	a = zero_one_gen(a,k) ;
	while (a) {
		if (is_verbose) print_numbers("test", a, k) ;
		sort_bitonic(a,k) ;
		if (is_verbose) print_numbers("sort", a, k) ;
		if (is_verbose) printf("\n") ;
		assert(test_array(a,k)==TRUE) ;
		a = zero_one_gen(NULL,0) ;
	}
	printf("success: zero-one test for size %d!\n\n", k) ;
	return 0 ;
}

int main(int argc, char * argv[]){
	
	char buffer[BUFFER_N] ;
	int ch ;
	int buf_n = BUFFER_N ;
	int numbers[MAX_NUMBERS] ;
	int n_num ;
	int is_filter = 0 ;
	int zo_n = 0 ;


	while ((ch = getopt(argc, argv, "vF")) != -1) {
		switch(ch) {
		case 'v':
			is_verbose = 1 ;
			break ;
		case 'F':
			is_filter = 1 ;
			break ;
		default:
			printf("usage: %s\n", USAGE_MESSAGE) ;
			return 0 ;
		}
	}

	argc -= optind;
	argv += optind;


	if (!is_filter) {
		if (argc!=1) {
			printf("usage: %s\n", USAGE_MESSAGE) ;
			return 0 ;
		}
		zo_n = atoi(argv[0]) ;
		assert(is_2power(zo_n)) ;
		return zero_one_test(zo_n) ;
	}
	assert(zo_n==0) ;


	n_num = 0 ;
	while( fgets(buffer, buf_n, stdin)) {
		char * s ;
		int i ;
		s = strtok(buffer, SEP_CHAR) ;
		while (s) {
			numbers[n_num++]= atoi(s) ;
			s = strtok(NULL, SEP_CHAR) ;
			assert(n_num<MAX_NUMBERS) ;
		}
	}
	n_num = pad_to_2power(numbers, n_num) ;
	assert(is_2power(n_num)) ;
	print_numbers("In", numbers, n_num) ;

	sort_bitonic(numbers, n_num) ;
	print_numbers("Out", numbers, n_num) ;
	printf("\n") ;
	return 0 ;
}



