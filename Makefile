CXX ?= g++ # This is the main compiler
# CXX := clang --analyze # and comment out the linker last line for sanity

CC ?= gcc

MPICXX ?= mpic++ -DOMPI_SKIP_MPICXX

ENV ?= gnu

NVCC ?= $(CUDA_PATH)/bin/nvcc

CUDA_PATH ?= /usr/local/cuda

SRCDIR := src
BINDIR := bin
BUILDDIR := build

THIS_FILE := $(lastword $(MAKEFILE_LIST))

CUDA_OBJS := $(BUILDDIR)/gather.o                     \
             $(BUILDDIR)/findneighbors.o              \
             $(BUILDDIR)/cudaDensity.o                \
             $(BUILDDIR)/cudaIAD.o                    \
             $(BUILDDIR)/cudaMomentumAndEnergyIAD.o

SEDOV_SOL_DIR := src/analytical_solutions/sedov_solution
SEDOV_SOL_CPP := $(SEDOV_SOL_DIR)/io.cpp              \
                 $(SEDOV_SOL_DIR)/sedov_solution.cpp  \
                 $(SEDOV_SOL_DIR)/main.cpp

RELEASE := -DNDEBUG
DEBUG := -D__DEBUG -D_GLIBCXX_DEBUG

# cuda architecture targets
SMS ?= 35 60 70 75
$(foreach sm,$(SMS),$(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))
GENCODE_FLAGS += -Wno-deprecated-gpu-targets

INC += -Isrc -Iinclude -Idomain/include -I$(CUDA_PATH)/include -I$(PGI_PATH)/include
CXXFLAGS += $(RELEASE)
NVCCFLAGS := -std=c++17 -O3 --expt-relaxed-constexpr -rdc=true $(GENCODE_FLAGS)
NVCCLDFLAGS := $(GENCODE_FLAGS) -rdc=true

CXXFLAGS += -O3 -Wall -Wextra -Wno-unknown-pragmas

ifeq ($(ENV),gnu)
	CXXFLAGS += -std=c++17 -fopenmp -fopenacc -march=native -mtune=native
endif

ifeq ($(ENV),pgi)
	CXXFLAGS += -std=c++17 -mp -dynamic -acc -ta=tesla,cc60 -mp=nonuma -Mcuda -g # -Minfo=accel # prints generated accel functions
endif

ifeq ($(ENV),cray)
	CXXFLAGS += -hstd=c++17 -homp -hacc -dynamic
endif

ifeq ($(ENV),intel)
	CXXFLAGS += -std=c++17 -qopenmp -dynamic
endif

ifeq ($(ENV),clang)
	CXXFLAGS += -march=native -std=c++17 -fopenmp
endif

TESTCASE ?= sedov
#TESTCASE ?= evrard

ifeq ($(TESTCASE),sedov)
	TESTCODE = src/sedov/sedov.cpp
	SOLCODE = $(SEDOV_SOL_CPP)
else ifeq ($(TESTCASE),evrard)
	TESTCASE_FLAGS = -DGRAVITY
	TESTCODE = src/evrard/evrard.cpp
endif

#omp:
#	@mkdir -p $(BINDIR)
#	$(info Linking the executable:)
#	$(CXX) $(CXXFLAGS) $(INC) $(TESTCASE_FLAGS) $(TESTCODE) -o $(BINDIR)/$@.app $(LIB)

mpi+omp:
	@mkdir -p $(BINDIR)
	$(info Linking the executable:)
	$(MPICXX) $(CXXFLAGS) $(INC) -DUSE_MPI $(TESTCASE_FLAGS) $(TESTCODE) -o $(BINDIR)/$@.app $(LIB)
ifdef SOLCODE
	make solution
endif
    
#omp+cuda: $(BUILDDIR)/cuda_no_mpi.o $(CUDA_OBJS)
#	@mkdir -p $(BINDIR)
#	$(info Linking the executable:)
#	$(NVCC) $(NVCCLDFLAGS) -DUSE_CUDA $(TESTCASE_FLAGS) -dlink -o cudalinked.o $(CUDA_OBJS) -lcudadevrt -lcudart
#	$(CXX) $(CXXFLAGS) -o $(BINDIR)/$@.app cudalinked.o $+ -L$(CUDA_PATH)/lib64 -lcudart -lcudadevrt
##	$(CXX) -o $(BINDIR)/$@.app $+ -L$(CUDA_PATH)/lib64 -lcudart -fopenmp

#omp+target:
#	@mkdir -p $(BINDIR)
#	$(info Linking the executable:)
#	$(CXX) $(CXXFLAGS) $(INC) -DUSE_OMP_TARGET $(TESTCASE_FLAGS) $(TESTCODE) -o $(BINDIR)/$@.app $(LIB)

mpi+omp+target:
	@mkdir -p $(BINDIR)
	$(info Linking the executable:)
	$(MPICXX) $(CXXFLAGS) $(INC) -DUSE_MPI -DUSE_OMP_TARGET $(TESTCASE_FLAGS) $(TESTCODE) -o $(BINDIR)/$@.app $(LIB)
ifdef SOLCODE
	make solution
endif

mpi+omp+acc:
	@mkdir -p $(BINDIR)
	$(info Linking the executable:)
	$(MPICXX) $(CXXFLAGS) $(INC) -DUSE_MPI -DUSE_STD_MATH_IN_KERNELS $(TESTCASE_FLAGS) -DUSE_ACC $(TESTCODE) -o $(BINDIR)/$@.app $(LIB)
ifdef SOLCODE
	make solution
endif

mpi+omp+cuda: $(BUILDDIR)/cuda_mpi.o $(CUDA_OBJS)
	@mkdir -p $(BINDIR)
	$(info Linking the executable:)
	$(NVCC) $(NVCCLDFLAGS) -dlink -o cudalinked.o $(CUDA_OBJS) -lcudadevrt -lcudart
	$(MPICXX) $(CXXFLAGS) -o $(BINDIR)/$@.app cudalinked.o $+ -L$(CUDA_PATH)/lib64 -lcudadevrt -lcudart
ifdef SOLCODE
	make solution
endif

solution:
	$(MPICXX) $(CXXFLAGS) $(INC) $(SOLCODE) -o $(BINDIR)/$(TESTCASE)_$@ $(LIB)

#all: omp mpi+omp omp+cuda mpi+omp+cuda omp+target mpi+omp+target mpi+omp+acc
all: mpi+omp mpi+omp+cuda mpi+omp+target mpi+omp+acc

$(BUILDDIR)/cuda_mpi.o: $(TESTCODE)
	@mkdir -p $(BUILDDIR)
	$(MPICXX) $(CXXFLAGS) $(INC) -DUSE_MPI -DUSE_CUDA $(TESTCASE_FLAGS) -o $@ -c $<

$(BUILDDIR)/cuda_no_mpi.o: $(TESTCODE)
	@mkdir -p $(BUILDDIR)
	$(CXX) $(CXXFLAGS) $(INC) -DUSE_CUDA $(TESTCASE_FLAGS) -o $@ -c $<

$(BUILDDIR)/%.o: include/sph/cuda/%.cu
	@mkdir -p $(BUILDDIR)
	$(NVCC) $(NVCCFLAGS) -DUSE_CUDA $(TESTCASE_FLAGS) $(INC) -c -o $@ $<
#	$(NVCC) $(NVCCFLAGS) $(INC) -DUSE_STD_MATH_IN_KERNELS -I$(CUDA_PATH)/include -L$(CUDA_PATH)/lib64 -c -o $@ $<

$(BUILDDIR)/%.o: domain/include/cstone/cuda/%.cu
	@mkdir -p $(BUILDDIR)
	$(NVCC) $(NVCCFLAGS) -DUSE_CUDA $(TESTCASE_FLAGS) $(INC) -c -o $@ $<

run_test:
#	@$(MAKE) -f $(THIS_FILE) omp
#	@$(MAKE) -f $(THIS_FILE) omp+cuda
	@$(MAKE) -f $(THIS_FILE) mpi+omp
	@$(MAKE) -f $(THIS_FILE) mpi+omp+cuda
	cd test/ && ./test_correctness.sh;

clean:
	$(info Cleaning...)
	$(RM) -rf $(BUILDDIR) $(BINDIR) cudalinked*.o

.PHONY: all clean
