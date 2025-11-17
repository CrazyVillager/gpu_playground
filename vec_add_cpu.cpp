#include<iostream>
#include<vector>
#include<chrono>
#include<random>
#include<cstdlib>

using namespace std;

int main(int argc, char **argv)
{
  if(argc < 2){
    cerr << "Usage: ./vec_add_cpu N \n";
    return 1;
  }

  size_t N = stoull(argv[1]);

  vector<double> a(N), b(N), y(N);

  mt19937_64 rng(1234);
  uniform_real_distribution<double> dist(0.0, 1.0);
  for(size_t i = 0; i < N; i++){
    a[i] = dist(rng);
    b[i] = dist(rng);
  }

  auto start = chrono::high_resolution_clock::now();

  for(size_t i = 0; i < N; i++){
    y[i] = a[i] + b[i];
  }

  auto end = chrono::high_resolution_clock::now();
  chrono::duration<double> elapsed = end - start;

  cout << "c[0] = " << y[0] << "\n";
  cout << "c[N - 1] = " << y[N - 1] << "\n";

  cout << "CPU time: " << elapsed.count() << "sec\n";

  return 0;
}
