#include <iostream>
#include <string>
#include <vector>
#include<chrono>
#include<random>
#include<cuda_runtime.h>

using namespace std;

__global__ void vecAddKernel(const double *a, const double *b, double *y, size_t N){
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < N){
    y[idx] = a[idx] + b[idx];
  }
}

int main(int argc, char **argv){
  if(argc < 3){
    cerr << "Usage: ./vec_add_gpu N blocksize\n";
    return 1;
  }

  size_t N = stoull(argv[1]);
  int blockSize = stoi(argv[2]);
  size_t bytes = N * sizeof(double);

  vector<double> h_a(N), h_b(N), h_y(N);

  mt19937_64 rng(1234);
  uniform_real_distribution<double> dist(0.0, 1.0);
  for(size_t i = 0; i < N; i++){
    h_a[i] = dist(rng);
    h_b[i] = dist(rng);
  }

  double *d_a, *d_b, *d_y;
  cudaMalloc(&d_a, bytes);
  cudaMalloc(&d_b, bytes);
  cudaMalloc(&d_y, bytes);

  cudaEvent_t start, afterH2D, afterKernel, end;
  cudaEventCreate(&start);
  cudaEventCreate(&afterH2D);
  cudaEventCreate(&afterKernel);
  cudaEventCreate(&end);

  cudaEventRecord(start);

  cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice);
  cudaEventRecord(afterH2D);

  // カーネル実行設定
  int numBlocks = (N + blockSize - 1) / blockSize;
  vecAddKernel<<<numBlocks, blockSize>>>(d_a, d_b, d_y, N);
  cudaEventRecord(afterKernel);
  cudaMemcpy(h_y.data(), d_y, bytes, cudaMemcpyDeviceToHost);
  cudaEventRecord(end);

  cudaDeviceSynchronize();

  float h2dTime, kernelTime, d2hTime, totalTime;
  cudaEventElapsedTime(&h2dTime, start, afterH2D);
  cudaEventElapsedTime(&kernelTime, afterH2D, afterKernel);
  cudaEventElapsedTime(&d2hTime, afterKernel, end);
  cudaEventElapsedTime(&totalTime, start, end);

  cout << "H2D: " << h2dTime / 1000 << "sec\n";
  cout << "Kernel: " << kernelTime / 1000 << "sec\n";
  cout << "D2H: " << d2hTime / 1000 << "sec\n";
  cout << "Total: " << totalTime / 1000 << "sec\n";
  // メモリ解放
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_y);

  return 0;
}
