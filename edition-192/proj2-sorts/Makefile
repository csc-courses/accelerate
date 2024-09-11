
#A= sm_30
A= sm_50

all:
	make test-o-e-sort

odd-even-sort: odd-even-sort.cu
	nvcc -arch $A -DN_ELEM=4 -o $@ $<

test-o-e-sort: odd-even-sort
	cat test-seq.txt | ./$< > test-seq.out
	diff test-seq.out test-seq.ref

bitonic-gpu: bitonic-sort-gpu.cu
	nvcc -arch $A -DN_ELEM=4 -o bitonic-sort-gpu bitonic-sort-gpu.cu

bitonic-gpu-debug: bitonic-sort-gpu.cu
	nvcc -arch $A0 -DN_ELEM=4 -DIS_VERBOSE=1 -o bitonic-sort-gpu bitonic-sort-gpu.cu

bitonic-sort-cpu: bitonic-sort-cpu.c
	cc -o $@ $<

test: bitonic-sort-cpu
	./$< 8
	cat test-seq.txt | ./$< -F > test-seq.out
	diff test-seq.out test-seq.ref

clean:
	-rm bitonic-sort-cpu bitonic-sort-gpu test-seq.out
	-rm odd-even-sort
