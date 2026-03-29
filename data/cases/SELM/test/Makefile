CC = nvcc
CCFLAG = -rdc=true -arch=sm_86 -O3 -std=c++17
LDFLAG = -rdc=true -arch=sm_86 -L$(HOME)/.local/lib -lmpi -lm  -llammps_mpi -lcufft -lcurand
INCLUDE =
EXE = selm_cuda
SRC = $(filter-out callback.cu, $(wildcard *.cu))
OBJ = $(patsubst %.cu,obj/%.o,$(SRC))
all: $(EXE)
$(EXE): $(OBJ)
	$(CC) $(OBJ) -o $@ $(LDFLAG)
obj/%.o: %.cu
	@mkdir -p obj
	$(CC) $(CCFLAG) $(INCLUDE) -c $< -o $@ 
clean:
	rm -f $(OBJ) $(EXE)
	rm -rf obj
.PHONY: all clean
