SRCDIR=src/
RES=resources/
#SRCDIR=.
MPICC=mpicc
NPROC=10
INPUT_TXT=test.txt
.PHONY: all clean old_version new_version run

all: new_version run_new

run_old: old_version
	mpiexec -n $(NPROC) ./prog_old $(RES)test.txt

run_new: new_version
	mpiexec -n $(NPROC) --disable-auto-cleanup ./prog_new $(RES)test.txt

old_version:
	$(MPICC) $(SRCDIR)old_version.c -o prog_old

new_version:
	$(MPICC) $(SRCDIR)new_version.c -o prog_new

clean:
	rm prog* output_file.txt
