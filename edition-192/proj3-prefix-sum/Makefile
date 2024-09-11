
#
# makefile for prefix-sum
# author: bjr
# date: 28 feb 2019
#

F= prefix-sum
BLOCKS=  2
THREADS= 8 
A= sm_70

all:
	make $F

$F: $F.cu
	nvcc -arch $A -DIS_VERBOSE=0 -o $@ $<

$F-cpu: $F-cpu.c
	cc -o $@ $<

test-small: $F
	cat test-seq.txt | ./$<  1 16
	cat test-seq.txt | ./$<  2 8 
	cat test-seq.txt | ./$<  4 4
	cat test-seq.txt | ./$<  8 2

test: $F
	cat test-seq.txt | ./$<  1 16
	cat test-seq.txt | ./$<  2 8 
	cat test-seq.txt | ./$<  4 4
	cat test-seq.txt | ./$<  8 2
	cat test-seq-4096.txt | ./$<  4 1024
	cat test-seq-4096.txt | ./$<  8 512
	cat test-seq-4096.txt | ./$<  16 256
	cat test-seq-4096.txt | ./$<  32 128
	cat test-seq-4096.txt | ./$<  64 64
	cat test-seq-4096.txt | ./$<  128 32
	cat test-seq-4096.txt | ./$<  256 16
	cat test-seq-4096.txt | ./$<  512 8
	cat test-seq-4096.txt | ./$<  1024 4

prof: $F $F-cpu
	cat test-seq-4096.txt | ./$<  4 1024
	cat test-seq-4096.txt | ./$<  256 16
	time ( cat test-seq-4096.txt | ./$F-cpu 4 1024 ) 

clean:
	-rm $F $F-cpu
