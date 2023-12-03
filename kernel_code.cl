//kernalFunction
__kernel void mat_mul_computation(const int computationNum,__global float* A, __global float* B, __global float* C) 
{
    int global_id_x = get_global_id(0);                                           
    int global_id_y = get_global_id(1);          
    int global_size_x = get_global_size(0);
                                 
    int global_index = global_id_y * global_size_x + global_id_x;      
    
    float result = 0.0f;
    for(int i = 0; i<computationNum; i++)                                                        
       result += A[global_index] + B[global_index];       

    C[global_index] = result;      
}     


//atomic_cmpxchg는 일반적으로 __global 메모리에 사용되는 것이 일반적입니다. __local 메모리에 대한 atomic 연산은 OpenCL에서 지원하지 않습니다. 
//__local 메모리는 워크 그룹 내에서 스레드 간에 데이터를 공유하는 용도로 사용되지만, 워크 그룹 간에 데이터를 공유할 때 atomic 연산을 수행하기 위한 지원은 제공되지 않습니다.
//따라서 atomic_cmpxchg를 __local 메모리에 사용하려고 하면 컴파일 오류가 발생할 것입니다. atomic_cmpxchg나 atomic 연산을 수행해야 하는 경우 주로 __global 메모리나 __private 메모리를 사용합니다.
inline void atomicAdd_g_f(volatile __global float *addr, float val)
{
union {
unsigned int u32;
float f32;
} next, expected, current;

current.f32 = *addr;
do {
expected.f32 = current.f32;
next.f32 = expected.f32 + val;
current.u32 = atomic_cmpxchg( (volatile __global unsigned int *)addr, expected.u32, next.u32);
} while( current.u32 != expected.u32 );
}


__kernel void Kmeans(__global const float* data, __global const float* centers,  __global int* cluster_indexes, 
 __global float* new_centers,  __global int* cluster_counts, int num_points, int num_dims, int k) 
 {   
    int gid = get_global_id(0);
    double min_dist = 999999;
    int closest_center = 0;
    for (int j = 0; j < k; ++j) 
    {
        double dist = 0.0;
        for (int col = 0; col < num_dims; ++col) 
        {
            double diff = data[gid * num_dims + col] - centers[j * num_dims + col];
            dist += diff * diff;
        }
        if (dist < min_dist) 
        {
            min_dist = dist;
            closest_center = j;
        }
    }
    cluster_indexes[gid] = closest_center;
    for (int col = 0; col < num_dims; ++col) 
        atomicAdd_g_f(&new_centers[closest_center * num_dims + col], data[gid * num_dims + col]);//Expensive....  
    atomic_inc(&cluster_counts[closest_center]);  
 }


    
