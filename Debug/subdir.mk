################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../cufftnd.cu \
../cumtrns.cu 

CPP_SRCS += \
../gpu_fftnd_tst.cpp 

C_SRCS += \
../cufftnd_.c 

O_SRCS += \
../cufftnd.o \
../cufftnd_.o \
../cumtrns.o 

OBJS += \
./cufftnd.o \
./cufftnd_.o \
./cumtrns.o \
./gpu_fftnd_tst.o 

CU_DEPS += \
./cufftnd.d \
./cumtrns.d 

CPP_DEPS += \
./gpu_fftnd_tst.d 

C_DEPS += \
./cufftnd_.d 


# Each subdirectory must supply rules for building sources it contributes
%.o: ../%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-9.2/bin/nvcc -G -g -O0 -gencode arch=compute_61,code=sm_61  -odir "." -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-9.2/bin/nvcc -G -g -O0 --compile --relocatable-device-code=false -gencode arch=compute_61,code=compute_61 -gencode arch=compute_61,code=sm_61  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

%.o: ../%.c
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-9.2/bin/nvcc -G -g -O0 -gencode arch=compute_61,code=sm_61  -odir "." -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-9.2/bin/nvcc -G -g -O0 --compile  -x c -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

%.o: ../%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-9.2/bin/nvcc -G -g -O0 -gencode arch=compute_61,code=sm_61  -odir "." -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-9.2/bin/nvcc -G -g -O0 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


