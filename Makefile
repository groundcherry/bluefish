################################################################################
################################# BLUEFISH-1.0 #################################
################################################################################
#
#   Copyright 2012 - 2014 Adam Sierakowski, The Johns Hopkins University
# 
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
# 
#       http://www.apache.org/licenses/LICENSE-2.0
# 
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
# 
#   Please contact the Johns Hopkins University to use Bluefish for
#   commercial and/or for-profit applications.
################################################################################

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ EDIT: DEPENDENCIES ~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
MPI_DIR =
HDF5_DIR =
CGNS_DIR =
CUDA_DIR =
CUDA_SDK_DIR =
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ EDIT: COMPILERS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
MPICC =
NVCC =
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

PREC = DOUBLE

SRC_DIR = src
SIM_DIR = sim

COPT = -std=c99 -pedantic -Wall -Wextra -fopenmp -D$(PREC)
LDINCS = -I $(MPI_DIR)/include -I $(CGNS_DIR)/include
LDLIBS = -lm -L $(HDF5_DIR)/lib -L $(CGNS_DIR)/lib -lcgns -lhdf5 

CUDAOPT = -arch=sm_30 -Xcompiler -fopenmp -m64 -D$(PREC)

CUDAINCS = -I $(CUDA_SDK_DIR)/common/inc
CUDALIBS = -L $(CUDA_DIR)/lib64 -lcudart

SRCC =	bluefish.c	\
	devices.c	\
	domain.c	\
	mpi_comm.c	\
	recorder.c	\
	vtk.c

SRCCUDA = cuda_comm.cu	\
	comm_kernel.cu

EXTRA = Makefile		\
	bluefish.h		\
	cuda_comm.h		\
	devices.h		\
	domain.h		\
	mpi_comm.h		\
	recorder.h		\
	vtk.h

# compile normally
all: COPT += -O2
all: CUDAOPT += -O2
all: bluefish

# compile with debug output
debug: COPT += -DDEBUG -g
debug: CUDAOPT += -DDEBUG -g -G
debug: bluefish

OBJS = $(addprefix $(SRC_DIR)/, $(addsuffix .o, $(basename $(SRCC))))
OBJSCUDA = $(addprefix $(SRC_DIR)/, $(addsuffix .o, $(basename $(SRCCUDA))))

$(OBJSCUDA):$(SRC_DIR)/%.o:$(SRC_DIR)/%.cu
	$(NVCC) $(CUDAOPT) -dc $< $(CUDAINCS) $(LDINCS) -o $@

$(OBJS):$(SRC_DIR)/%.o:$(SRC_DIR)/%.c
	$(MPICC) $(COPT) -c $< $(LDINCS) -o $@

$(SRC_DIR)/bblib.o:$(OBJSCUDA)
	$(NVCC) $(CUDAOPT) -dlink $+ -o $(SRC_DIR)/bblib.o $(CUDALIBS)

bluefish: $(OBJSCUDA) $(SRC_DIR)/bblib.o $(OBJS) 
	$(MPICC) $(COPT) -o $(SIM_DIR)/bluefish $+ $(LDLIBS) $(CUDALIBS) -lstdc++

clean:
	rm -f $(SRC_DIR)/*.o $(SIM_DIR)/bluefish
