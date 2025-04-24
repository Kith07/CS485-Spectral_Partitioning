
all: compile run_all

compile:
	g++ -I ./Eigen pr.cpp -O2 -o pr1

run_all: run_4 run_100 run_1000 run_10000

run_4:
	./pr1.exe input_4.txt

run_100:
	./pr1.exe input_100.txt

run_1000:
	./pr1.exe input_1000.txt

run_10000:
	./pr1.exe input_10000.txt