SRCDIR=src/
MPICC=mpiCC
NPROC=16
THREADS = 2
.PHONY: all clean omp_version seq_version run_seq run_omp

all: seq_version run_seq

run_omp: omp_version
	mpirun -n $(NPROC) -genv OMP_NUM_THREADS=$(THREADS) -genv I_MPI_PIN_DOMAIN=omp ./prog_omp_version

run_seq: seq_version
	mpiexec -n $(NPROC) ./prog_seq_version

omp_version:
	$(MPICC) $(SRCDIR)mpi_scan_parallel.cpp -o prog_omp_version -fopenmp

seq_version:
	$(MPICC) $(SRCDIR)mpi_scan_seq.cpp -o prog_seq_version

clean:
	rm prog*
