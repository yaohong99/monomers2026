CC = mpicxx
CCFLAG = -g -O3 -std=c++17
LAMMPS_PATH = $(HOME)/lammps-22Jul2025/src
LDFLAG = -L$(LAMMPS_PATH) -llammps_mpi -lfftw3_mpi -lfftw3
INCLUDE = -I$(LAMMPS_PATH)
EXE = selm_mpi
SRC = $(wildcard *.cpp)
OBJ = $(patsubst %.cpp,obj/%.o,$(SRC))
all: $(EXE)
$(EXE): $(OBJ)
	$(CC) $(OBJ) -o $@ $(LDFLAG)
obj/%.o: %.cpp
	@mkdir -p obj
	$(CC) $(CCFLAG) $(INCLUDE) -c $< -o $@ 
clean:
	rm -f $(OBJ) $(EXE)
	rm -rf obj
.PHONY: all clean
