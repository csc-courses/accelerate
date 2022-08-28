#
# accelerate! GPU
# csc596 (231) fall 2022-2023
#
# proj1 makefile
# last-update:
#	28 aug 2022 -bjr: 
#
#

H= cryptanthus
H= lutheria
H= androlepis
H= orthophytum

L=/usr/local/cuda/bin
GIT_NAME= __YOUR_GITHUB_NAME__

run: hello
	./hello

test: 
	make clean
	make hello
	./hello > hello.out
	git add hello.out
	git commit -m "proj1 submit"
	git push

git-config:
	git config --global user.name ${GIT_NAME}
	git config --global credential.helper store

hello:
	$L/nvcc hello.cu -o hello

login:
	ssh $H

clean:
	-rm hello hello.out
