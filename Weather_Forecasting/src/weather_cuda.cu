/* Two NVIDIA CUDA implementations for eight REAL weather CSV measurements.
 * Global atomic: every valid cell updates 3 global aggregations.
 * Shared memory: each 256-thread block reduces by parameter before atomic adds.
 * Both produce mean, standard deviation, and per-parameter valid counts.
 * NaNs are skipped. No synthetic weather generation. GTX 1650 sm_75 supported.
 */
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#define P 8
#define BLOCK 256
#define CUDA_OK(call) do { cudaError_t err=(call); if (err!=cudaSuccess) { \
    fprintf(stderr,"CUDA %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(err)); exit(3); } } while(0)

__global__ void kernel_atomic(const float *data, double *sum, double *square,
                              unsigned long long *count, unsigned long long elements) {
    unsigned long long idx=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;
    if (idx>=elements) return;
    float v=data[idx];
    if (!isfinite(v)) return;
    int p=(int)(idx%P);
    double d=(double)v;
    atomicAdd(&sum[p],d);
    atomicAdd(&square[p],d*d);
    atomicAdd(&count[p],1ULL);
}
__global__ void kernel_shared(const float *data, double *sum, double *square,
                              unsigned long long *count, unsigned long long elements) {
    __shared__ double local_sum[BLOCK];
    __shared__ double local_square[BLOCK];
    __shared__ unsigned int local_count[BLOCK];
    unsigned int t=threadIdx.x;
    unsigned long long idx=(unsigned long long)blockIdx.x*BLOCK+t;
    double d=0.0;
    bool valid=idx<elements && isfinite(data[idx]);
    if (valid) d=(double)data[idx];
    local_sum[t]=valid?d:0.0;
    local_square[t]=valid?d*d:0.0;
    local_count[t]=valid?1U:0U;
    __syncthreads();
    /* The 8 metric columns repeat every 8 entries; reductions preserve idx % 8. */
    for (unsigned int stride=BLOCK/2;stride>=P;stride>>=1) {
        if (t<stride) {
            local_sum[t]+=local_sum[t+stride];
            local_square[t]+=local_square[t+stride];
            local_count[t]+=local_count[t+stride];
        }
        __syncthreads();
    }
    if (t<P && local_count[t]) {
        atomicAdd(&sum[t],local_sum[t]);
        atomicAdd(&square[t],local_square[t]);
        atomicAdd(&count[t],(unsigned long long)local_count[t]);
    }
}
static bool positive_integer(const char *text,unsigned long long *output) {
    if (!text || !*text || *text=='-') return false;
    char *end=nullptr; unsigned long long v=strtoull(text,&end,10);
    if (!end || *end || v==0 || v>200000000ULL) return false;
    *output=v; return true;
}
int main(int argc,char **argv) {
    const char *input=nullptr, *variant="shared";
    unsigned long long rows=0, passes=10;
    for (int i=1;i<argc;++i) {
        if (!strcmp(argv[i],"--input") && i+1<argc) input=argv[++i];
        else if (!strcmp(argv[i],"--rows") && i+1<argc && positive_integer(argv[i+1],&rows)) ++i;
        else if (!strcmp(argv[i],"--passes") && i+1<argc && positive_integer(argv[i+1],&passes)) ++i;
        else if (!strcmp(argv[i],"--variant") && i+1<argc) variant=argv[++i];
        else { fprintf(stderr,"Usage: weather_cuda --input PATH --rows N --passes N --variant atomic|shared\n");return 2; }
    }
    if (!input || !rows || passes>100000 || (strcmp(variant,"atomic") && strcmp(variant,"shared"))) {
        fprintf(stderr,"Missing or invalid CUDA args\n");return 2;
    }
    int devices=0; CUDA_OK(cudaGetDeviceCount(&devices));
    if (devices<1) { fprintf(stderr,"No CUDA GPU found\n");return 3; }
    cudaDeviceProp prop; CUDA_OK(cudaGetDeviceProperties(&prop,0));
    if (prop.major<6) {fprintf(stderr,"GPU needs compute capability 6+ for double atomicAdd\n");return 3;}
    unsigned long long elements=rows*P;
    size_t bytes=(size_t)elements*sizeof(float);
    float *host=(float*)malloc(bytes);
    if (!host) {fprintf(stderr,"Cannot allocate host memory\n");return 3;}
    FILE *f=fopen(input,"rb");
    if (!f) {fprintf(stderr,"Missing real-CSV-derived binary input: %s\n",input);free(host);return 3;}
    if (fread(host,sizeof(float), (size_t)elements,f)!=(size_t)elements) {
        fprintf(stderr,"CSV-derived file has fewer rows than requested\n");fclose(f);free(host);return 3;
    }
    fclose(f);
    float *device_data=nullptr;
    double *device_sum=nullptr,*device_square=nullptr;
    unsigned long long *device_count=nullptr;
    CUDA_OK(cudaMalloc((void**)&device_data,bytes));
    CUDA_OK(cudaMalloc((void**)&device_sum,P*sizeof(double)));
    CUDA_OK(cudaMalloc((void**)&device_square,P*sizeof(double)));
    CUDA_OK(cudaMalloc((void**)&device_count,P*sizeof(unsigned long long)));
    cudaEvent_t start,finish;
    CUDA_OK(cudaEventCreate(&start));CUDA_OK(cudaEventCreate(&finish));
    auto t0=std::chrono::steady_clock::now();
    CUDA_OK(cudaMemcpy(device_data,host,bytes,cudaMemcpyHostToDevice));
    CUDA_OK(cudaEventRecord(start));
    unsigned long long num_blocks=(elements+BLOCK-1)/BLOCK;
    if (num_blocks>2147483647ULL) {fprintf(stderr,"CUDA grid too large\n");return 3;}
    dim3 grid((unsigned int)num_blocks), block(BLOCK);
    for (unsigned long long step=0;step<passes;++step) {
        CUDA_OK(cudaMemset(device_sum,0,P*sizeof(double)));
        CUDA_OK(cudaMemset(device_square,0,P*sizeof(double)));
        CUDA_OK(cudaMemset(device_count,0,P*sizeof(unsigned long long)));
        if (!strcmp(variant,"atomic")) kernel_atomic<<<grid,block>>>(device_data,device_sum,device_square,device_count,elements);
        else kernel_shared<<<grid,block>>>(device_data,device_sum,device_square,device_count,elements);
        CUDA_OK(cudaGetLastError());
    }
    CUDA_OK(cudaEventRecord(finish));CUDA_OK(cudaEventSynchronize(finish));
    float compute_ms=0;
    CUDA_OK(cudaEventElapsedTime(&compute_ms,start,finish));
    double sum[P],square[P];
    unsigned long long count[P];
    CUDA_OK(cudaMemcpy(sum,device_sum,sizeof(sum),cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(square,device_square,sizeof(square),cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(count,device_count,sizeof(count),cudaMemcpyDeviceToHost));
    auto t1=std::chrono::steady_clock::now();
    double total_ms=std::chrono::duration<double,std::milli>(t1-t0).count();
    printf("{\"method\":\"cuda_%s\",\"rows\":%llu,\"passes\":%llu,\"threads\":0,\"compute_ms\":%.6f,\"total_ms\":%.6f,\"gpu_name\":\"%s\",\"counts\":[",
           variant,rows,passes,(double)compute_ms,total_ms,prop.name);
    for(int p=0;p<P;++p) printf("%s%llu",p?",":"",count[p]);
    printf("],\"means\":[");
    for(int p=0;p<P;++p) {
        if(p) printf(",");
        if(count[p]) printf("%.12g",sum[p]/(double)count[p]);
        else printf("null");
    }
    printf("],\"stddevs\":[");
    for(int p=0;p<P;++p) {
        if(p) printf(",");
        if(count[p]) {
            double mean=sum[p]/(double)count[p];
            double variance=square[p]/(double)count[p]-mean*mean;
            printf("%.12g",sqrt(fmax(variance,0.0)));
        } else printf("null");
    }
    printf("]}\n");
    CUDA_OK(cudaEventDestroy(start));CUDA_OK(cudaEventDestroy(finish));
    CUDA_OK(cudaFree(device_data)); CUDA_OK(cudaFree(device_sum));
    CUDA_OK(cudaFree(device_square));CUDA_OK(cudaFree(device_count));
    free(host);
    return 0;
}
