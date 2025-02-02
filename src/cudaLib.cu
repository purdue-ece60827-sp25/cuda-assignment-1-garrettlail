
#include "cudaLib.cuh"
#include "lab1.cuh"

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort)
{
	if (code != cudaSuccess) 
	{
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

__global__ 
void saxpy_gpu (float* x, float* y, float scale, int size) {
	//	Insert GPU SAXPY kernel code here
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size)
	{
		y[i] = scale * x[i] + y[i];
	}
}

int runGpuSaxpy(int vectorSize) {

	std::cout << "Hello GPU Saxpy!\n";

	int sizeInB = vectorSize * sizeof(float);
	cudaError_t error = cudaSuccess;
	float scale = 0.0f;
	vectorInit(&scale, 1);

	// Allocate memory for the CPU
	float *cpu_x, *cpu_y, *cpu_gpu_result;
	cpu_x = (float *) malloc(sizeInB);
	cpu_y = (float *) malloc(sizeInB);
	cpu_gpu_result = (float *) malloc(sizeInB);

	if ((cpu_x == NULL) || (cpu_y == NULL) || (cpu_gpu_result == NULL))
	{
		printf("Allocation of CPU array failed... Exiting!");
		return -1;
	}

	// Initialize the CPU input vectors with random numbers
	vectorInit(cpu_x, vectorSize);
	vectorInit(cpu_y, vectorSize);

	// Initialize memory for the GPU
	float *gpu_x, *gpu_y;
	error = cudaMalloc(&gpu_x, sizeInB);
	if (error != cudaSuccess)
	{
		printf("Allocation of GPU array failed... Exiting!");
		return -1;
	}

	error = cudaMalloc(&gpu_y, sizeInB);
	if (error != cudaSuccess)
	{
		printf("Allocation of GPU array failed... Exiting!");
		return -1;
	}

	// Perform memory transfers from CPU to GPU for operands
	error = cudaMemcpy(gpu_x, cpu_x, sizeInB, cudaMemcpyHostToDevice);
	if (error != cudaSuccess)
	{
		printf("Memory transfer from CPU -> GPU failed... Exiting!");
		return -1;
	}

	error = cudaMemcpy(gpu_y, cpu_y, sizeInB, cudaMemcpyHostToDevice);
	if (error != cudaSuccess)
	{
		printf("Memory transfer from CPU -> GPU failed... Exiting!");
		return -1;
	}

	// Launch the SAXPY kernel with 256 threads per Thread Block
	int total_thread_blocks = vectorSize / 256 + 1;
	auto tStart = std::chrono::high_resolution_clock::now();
	saxpy_gpu<<<total_thread_blocks, 256>>>(gpu_x, gpu_y, scale, vectorSize);
	auto tEnd= std::chrono::high_resolution_clock::now();
	cudaDeviceSynchronize();

	// Copy results from GPU -> CPU
	error = cudaMemcpy(cpu_gpu_result, gpu_y, sizeInB, cudaMemcpyDeviceToHost);
	if (error != cudaSuccess)
	{
		printf("Memory transfer from GPU -> CPU failed... Exiting!");
		return -1;
	}

	// Compare results
	int result = verifyVector(cpu_x, cpu_y, cpu_gpu_result, scale, vectorSize);
	printf("Number of Errors: %d\n", result);

	// Print results
	#ifndef DEBUG_PRINT_DISABLE 
		printf("\n Adding vectors : \n");
		printf(" scale = %f\n", scale);
		printf(" a = { ");
		for (int i = 0; i < 5; ++i) {
			printf("%3.4f, ", cpu_x[i]);
		}
		printf(" ... }\n");
		printf(" b = { ");
		for (int i = 0; i < 5; ++i) {
			printf("%3.4f, ", cpu_y[i]);
		}
		printf(" ... }\n");

		printf(" c = { ");
		for (int i = 0; i < 5; ++i) {
			printf("%3.4f, ", cpu_gpu_result[i]);
		}
		printf(" ... }\n");
	#endif

	cudaFree(gpu_x);
	cudaFree(gpu_y);
	free(cpu_x);
	free(cpu_y);
	free(cpu_gpu_result);

	std::chrono::duration<double> time_span = (tEnd- tStart);
	std::cout << "It took " << time_span.count() << " seconds.";

	return 0;
}

/* 
 Some helpful definitions

 generateThreadCount is the number of threads spawned initially. Each thread is responsible for sampleSize points. 
 *pSums is a pointer to an array that holds the number of 'hit' points for each thread. The length of this array is pSumSize.

 reduceThreadCount is the number of threads used to reduce the partial sums.
 *totals is a pointer to an array that holds reduced values.
 reduceSize is the number of partial sums that each reduceThreadCount reduces.

*/

// 0.17
__global__
void generatePoints (uint64_t * pSums, uint64_t pSumSize, uint64_t sampleSize) {
	int tid = blockDim.x * blockIdx.x + threadIdx.x;

	// Setup RNG
	curandState_t rng;
	curand_init(clock64(), tid, 0, &rng);

    // Get a new random value
	double x, y;
	uint64_t sum = 0;
	for (int i = 0; i < sampleSize; i++)
	{
		x = curand_uniform(&rng);
		y = curand_uniform(&rng);
		sum += int(x * x + y * y) == 0;
	}

	if (tid < pSumSize)
	{
		pSums[tid] = sum;
	}
}

__global__ 
void reduceCounts (uint64_t * pSums, uint64_t * totals, uint64_t pSumSize, uint64_t reduceSize) {
	int tid = blockDim.x * blockIdx.x + threadIdx.x;

	uint64_t sum = 0;
	for (int i = 0; i < reduceSize; i++)
	{
		int index = tid * reduceSize + i;
		if (index < pSumSize)
		{
			sum += pSums[index];
		}
	}

	totals[tid] = sum;
	return;
}

int runGpuMCPi (uint64_t generateThreadCount, uint64_t sampleSize, 
	uint64_t reduceThreadCount, uint64_t reduceSize) {

	//  Check CUDA device presence
	int numDev;
	cudaGetDeviceCount(&numDev);
	if (numDev < 1) {
		std::cout << "CUDA device missing!\n";
		return -1;
	}

	auto tStart = std::chrono::high_resolution_clock::now();
		
	float approxPi = estimatePi(generateThreadCount, sampleSize, 
		reduceThreadCount, reduceSize);
	
	std::cout << std::setprecision(10);
	std::cout << "Estimated Pi = " << approxPi << "\n";

	auto tEnd= std::chrono::high_resolution_clock::now();

	std::chrono::duration<double> time_span = (tEnd- tStart);
	std::cout << "It took " << time_span.count() << " seconds.";

	return 0;
}

double estimatePi(uint64_t generateThreadCount, uint64_t sampleSize, 
	uint64_t reduceThreadCount, uint64_t reduceSize) {
	
	double approxPi = 0;

	// Allocate generateThreadCount number of pSums on GPU
	// Pass off to point generation

	// allocate reduceThreadCount number of totals on GPU & CPU
	// pass to reduce function
	// return reduced array to cpu

	// for gpu: calculate the number of sums to reduce based on thread index and if you are over end of array (cuz may be remainder)

	// generateThreadCount / reduceSize = # of points remaining in totals

	// The number of partial sums to be computed...
	const int numPartialSums = generateThreadCount;

	// The number of reduced sums to be computed...
	const int numReducedSums = reduceThreadCount;

	// The leftover partial sums. Should be 0.
	const int remPartialSums = numPartialSums - numReducedSums * reduceSize;

	if ((generateThreadCount <= 0) || (sampleSize <= 0) || (reduceThreadCount <= 0) || (reduceSize <= 0))
	{
		printf("Requested generation or reduction of negative or zero size... Exiting!\n");
		return -1;
	}

	if ((numReducedSums * reduceSize) > numPartialSums)
	{
		printf("Requested more reductions than generations... Exiting!\n");
		return -1;
	}

	// Allocate the final summation vector on the CPU
	uint64_t* cpu_pSums = (uint64_t *) malloc((numReducedSums + remPartialSums) * sizeof(uint64_t));
	if (cpu_pSums == NULL)
	{
		printf("Allocation of CPU array failed... Exiting!\n");
		return -1;
	}

	// Allocate the partial sums vector and reduced sums vectors on GPU
	cudaError_t error = cudaSuccess;
	uint64_t *gpu_pSums, *gpu_rSums;

	error = cudaMalloc(&gpu_pSums, numPartialSums * sizeof(uint64_t));
	if (error != cudaSuccess)
	{
		printf("Allocation of GPU array failed... Exiting!\n");
		return -1;
	}

	error = cudaMalloc(&gpu_rSums, numReducedSums * sizeof(uint64_t));
	if (error != cudaSuccess)
	{
		printf("Allocation of GPU array failed... Exiting!\n");
		return -1;
	}

	// Initialize CUDA kernel to generate the partial sums
	int total_thread_blocks = generateThreadCount / 256 + 1;
	generatePoints<<<total_thread_blocks, 256>>>(gpu_pSums, numPartialSums, sampleSize);

	// Initialize the CUDA kernel to sum the partial results
	total_thread_blocks = reduceThreadCount / 256 + 1;
	reduceCounts<<<total_thread_blocks, 256>>>(gpu_pSums, gpu_rSums, numPartialSums, reduceSize);
	cudaDeviceSynchronize();

	// Transfer results from GPU to CPU
	error = cudaMemcpy(cpu_pSums, gpu_rSums, numReducedSums * sizeof(uint64_t), cudaMemcpyDeviceToHost);
	if (error != cudaSuccess)
	{
		printf("Memory transfer from GPU -> CPU failed... Exiting!\n");
		return -1;
	}

	if (remPartialSums != 0)
	{
		printf("Transfering %dB from gpu[%d] to cpu[%d]\n", remPartialSums * sizeof(uint64_t), numReducedSums * reduceSize, numReducedSums);
		printf("Sizeof(gpu): %d, Sizeof(cpu): %d", numPartialSums * sizeof(uint64_t), (numReducedSums + remPartialSums) * sizeof(uint64_t));
		error = cudaMemcpy(&cpu_pSums[numReducedSums], &gpu_pSums[numReducedSums * reduceSize], 
				remPartialSums * sizeof(uint64_t), cudaMemcpyDeviceToHost);
		if (error != cudaSuccess)
		{
			std::cerr << "cudaMemcpy failed: " << cudaGetErrorString(error) << std::endl;
			printf("Memory transfer from GPU -> CPU failed... Exiting!\n");
			return -1;
		}
	}

	// Sum final results
	uint64_t totalHitCount = 0;
	for (int i = 0; i < (numReducedSums + remPartialSums); i++)
	{
		totalHitCount += cpu_pSums[i];
	}

	// De-allocate memory
	free(cpu_pSums);
	cudaFree(gpu_pSums);
	cudaFree(gpu_rSums);

	//	Calculate Pi
	printf("Hit Count: %ld\n", totalHitCount);
	approxPi = (double)totalHitCount / (numPartialSums * sampleSize);
	approxPi = approxPi * 4.0f;
	return approxPi;
}
