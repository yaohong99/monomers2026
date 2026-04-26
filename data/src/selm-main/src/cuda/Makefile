CC = nvcc
CCFLAG = -rdc=true -arch=sm_75 -O3 -std=c++17
LAMMPS_PATH = $(HOME)/lammps-22Jul2025/src
LDFLAG = -rdc=true -arch=sm_75 -L$(LAMMPS_PATH) -lmpi  -llammps_mpi -lcufft -lcurand
INCLUDE = -I$(LAMMPS_PATH)
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
