#include <cstdint>
#include <cuda_runtime.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iostream>
#include <sstream>
#include <stdint.h>

namespace needle {
namespace cuda {

#define BASE_THREAD_NUM 256

#define TILE 4
#define V TILE
#define L (16 * V)
#define S 16

typedef float scalar_t;
const size_t ELEM_SIZE = sizeof(scalar_t);

struct CudaArray {
  CudaArray(const size_t size) {
    cudaError_t err = cudaMalloc(&ptr, size * ELEM_SIZE);
    if (err != cudaSuccess)
      throw std::runtime_error(cudaGetErrorString(err));
    this->size = size;
  }
  ~CudaArray() { cudaFree(ptr); }
  size_t ptr_as_int() { return (size_t)ptr; }

  scalar_t *ptr;
  size_t size;
};

struct CudaDims {
  dim3 block, grid;
};

CudaDims CudaOneDim(size_t size) {
  /**
   * Utility function to get cuda dimensions for 1D call
   */
  CudaDims dim;
  size_t num_blocks = (size + BASE_THREAD_NUM - 1) / BASE_THREAD_NUM;
  dim.block = dim3(BASE_THREAD_NUM, 1, 1);
  dim.grid = dim3(num_blocks, 1, 1);
  return dim;
}

#define MAX_VEC_SIZE 8
struct CudaVec {
  uint32_t size;
  int32_t data[MAX_VEC_SIZE];
};

CudaVec VecToCuda(const std::vector<int32_t> &x) {
  CudaVec shape;
  if (x.size() > MAX_VEC_SIZE)
    throw std::runtime_error("Exceeded CUDA supported max dimesions");
  shape.size = x.size();
  for (size_t i = 0; i < x.size(); i++) {
    shape.data[i] = x[i];
  }
  return shape;
}

////////////////////////////////////////////////////////////////////////////////
// Fill call
////////////////////////////////////////////////////////////////////////////////

__global__ void FillKernel(scalar_t *out, scalar_t val, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size)
    out[gid] = val;
}

void Fill(CudaArray *out, scalar_t val) {
  CudaDims dim = CudaOneDim(out->size);
  FillKernel<<<dim.grid, dim.block>>>(out->ptr, val, out->size);
}

////////////////////////////////////////////////////////////////////////////////
// Compact and setitem cals
////////////////////////////////////////////////////////////////////////////////

// Utility function to convert contiguous index i to memory location from
// strides
__device__ size_t compute_index(size_t i, CudaVec shape, CudaVec strides,
                                size_t offset) {
  /**
   * Given an index i into a compact array (shape), computes the corresponding
   * index into a non-compact array (shape, strides, offset).
   *
   * Args:
   *   i: index into a compact array
   *   shape: shape of both the compact & non-compact arrays
   *   strides: stride of non-compact array
   *   offset: offset of non-compact array
   *
   * Returns:
   *   j: index into non-compact array
   */
  assert(shape.size == strides.size &&
         "compute_index: shape, strides mismatch");

  size_t j = offset;
  for (int32_t k = shape.size - 1; k >= 0; k--) {
    j += (i % shape.data[k]) * strides.data[k];
    i /= shape.data[k];
  }
  assert(i == 0);

  return j;
}

__global__ void CompactKernel(const scalar_t *a, scalar_t *out, size_t size,
                              CudaVec shape, CudaVec strides, size_t offset) {
  /**
   * The CUDA kernel for the compact operation.  This should effectively map a
   * single entry in the non-compact input a, to the corresponding item (at
   * location gid) in the compact array out.
   *
   * Args:
   *   a: CUDA pointer to a array
   *   out: CUDA pointer to out array
   *   size: size of out array
   *   shape:
   *     vector of shapes of a and out arrays (of type CudaVec, for past passing
   *     to CUDA kernel)
   *   strides: vector of strides of out (a?) array
   *   offset: offset of out (a?) array
   */
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

  /// BEGIN SOLUTION
  if (gid < size) {
    size_t j = compute_index(gid, shape, strides, offset);
    out[gid] = a[j];
  }
  /// END SOLUTION
}

void Compact(const CudaArray &a, CudaArray *out, std::vector<int32_t> shape,
             std::vector<int32_t> strides, size_t offset) {
  /**
   * Compact an array in memory.  Unlike the C++ version, in CUDA this will
   * primarily call the relevant CUDA kernel.  In this case, we illustrate how
   * you should set this up (i.e., we give you the code for this fuction, and
   * also the prototype for the CompactKernel() function).  For the functions
   * after this, however, you'll need to define these kernels as you see fit to
   * execute the underlying function.
   *
   * Args:
   *   a: non-compact represntation of the array, given as input
   *   out: compact version of the array to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *a* array (not out, which has compact strides)
   *   offset: offset of the *a* array (not out, which has zero offset, being
   *     compact)
   */

  // Nothing needs to be added here
  CudaDims dim = CudaOneDim(out->size);
  CompactKernel<<<dim.grid, dim.block>>>(
      a.ptr, out->ptr, out->size, VecToCuda(shape), VecToCuda(strides), offset);
}

__global__ void EwiseSetItemKernel(const scalar_t *a, scalar_t *out,
                                   size_t size, CudaVec shape, CudaVec strides,
                                   size_t offset) {
  /**
   * The CUDA kernel for the EwiseSetitem operation.
   *
   * Args:
   *   a: CUDA pointer to a compact array
   *   out: CUDA pointer to non-compact array
   *   size: size of out array
   *   shape: shapes of a and out arrays
   *   strides: strides of out array
   *   offset: offset of out array
   */
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) {
    size_t j = compute_index(gid, shape, strides, offset);
    out[j] = a[gid];
  }
}

void EwiseSetitem(const CudaArray &a, CudaArray *out,
                  std::vector<int32_t> shape, std::vector<int32_t> strides,
                  size_t offset) {
  /**
   * Set items in a (non-compact) array using CUDA.  You will most likely want
   * to implement a EwiseSetitemKernel() function, similar to those above, that
   * will do the actual work.
   *
   * Args:
   *   a: _compact_ array whose items will be written to out
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *out* array (not a, which has compact strides)
   *   offset: offset of the *out* array (not a, which has zero offset, being
   *   compact)
   */
  /// BEGIN SOLUTION
  CudaDims dim = CudaOneDim(out->size);
  EwiseSetItemKernel<<<dim.grid, dim.block>>>(
      a.ptr, out->ptr, a.size, VecToCuda(shape), VecToCuda(strides), offset);
  /// END SOLUTION
}

__global__ void ScalarSetItemKernel(const scalar_t val, scalar_t *out,
                                    size_t size, CudaVec shape, CudaVec strides,
                                    size_t offset) {
  /**
   * The CUDA kernel for the ScalarSetitem operation.
   *
   * Args:
   *   val: pointer to a constant value to set out to
   *   out: CUDA pointer to non-compact array
   *   size: size of out array
   *   shape: shape of out array
   *   strides: strides of out array
   *   offset: offset of out array
   */
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) {
    size_t j = compute_index(gid, shape, strides, offset);
    out[j] = val;
  }
}

void ScalarSetitem(size_t size, scalar_t val, CudaArray *out,
                   std::vector<int32_t> shape, std::vector<int32_t> strides,
                   size_t offset) {
  /**
   * Set items is a (non-compact) array
   *
   * Args:
   *   size:
   *     number of elements to write in out array (note that this will not
   *     be the same as out.size, because out is a non-compact subset array);
   *     it _will_ be the same as the product of items in shape, but covenient
   *     to just pass it here.
   *   val: scalar value to write to out
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension of out
   *   strides: strides of the out array
   *   offset: offset of the out array
   */
  /// BEGIN SOLUTION
  CudaDims dim = CudaOneDim(size);
  ScalarSetItemKernel<<<dim.grid, dim.block>>>(
      val, out->ptr, size, VecToCuda(shape), VecToCuda(strides), offset);
  /// END SOLUTION
}

////////////////////////////////////////////////////////////////////////////////
// Elementwise and scalar operations
////////////////////////////////////////////////////////////////////////////////

__global__ void EwiseAddKernel(const scalar_t *a, const scalar_t *b,
                               scalar_t *out, size_t size) {
  // Calculate the global index of the thread.
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size)
    out[gid] = a[gid] + b[gid];
}

void EwiseAdd(const CudaArray &a, const CudaArray &b, CudaArray *out) {
  /**
   * Add together two CUDA arrays.
   * Args:
   *   a: Input array 'a' to be added
   *   b: Input array 'b' to be added
   *   out: Output array to store the result of 'a + b'
   */
  CudaDims dim = CudaOneDim(out->size);

  // Kernel will execute on 'dim.grid' blocks, each containing 'dim.block'
  // threads.
  EwiseAddKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}

__global__ void ScalarAddKernel(const scalar_t *a, scalar_t val, scalar_t *out,
                                size_t size) {
  // Calculate the global index of the thread.
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size)
    out[gid] = a[gid] + val;
}

void ScalarAdd(const CudaArray &a, scalar_t val, CudaArray *out) {
  /**
   * Add a scalar value to every element of a CUDA array.
   * Args:
   *   a: Input array 'a'
   *   val: Scalar value to be added
   *   out: Output array to store the result of 'a + val'
   */
  CudaDims dim = CudaOneDim(out->size);

  // Launch the ScalarAddKernel that will add the scalar 'val' to each element
  // of array 'a', and store the result in array 'out'.
  ScalarAddKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}

/**
 * In the code the follows, use the above template to create analogous
 * elementwise and and scalar operators for the following functions.  See the
 * numpy backend for examples of how they should work.
 *  - EwiseMul, ScalarMul
 *  - EwiseDiv, ScalarDiv
 *  - ScalarPower
 *  - EwiseMaximum, ScalarMaximum
 *  - EwiseEq, ScalarEq
 *  - EwiseGe, ScalarGe
 *  - EwiseLog
 *  - EwiseExp
 *  - EwiseTanh
 *
 * If you implement all these naively, there will be a lot of repeated code, so
 * you are welcome (but not required), to use macros or templates to define
 * these functions (however you want to do so, as long as the functions match
 * the proper) signatures above.
 */

/// BEGIN TEMPLATES
template <typename Op>
__global__ void EwiseBinaryOpKernel(const scalar_t *a, const scalar_t *b,
                                    scalar_t *out, size_t size, Op op) {
  // Calculate the global index of the thread — assuming we called CudaOneDim()
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;

  for (size_t i = gid; i < size; i += stride) {
    out[i] = op(a[gid], b[gid]);
  }
}

template <typename Op>
__global__ void ScalarBinaryOpKernel(const scalar_t *a, scalar_t val,
                                     scalar_t *out, size_t size, Op op) {
  // Calculate the global index of the thread — assuming we called CudaOneDim()
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;

  for (size_t i = gid; i < size; i += stride) {
    out[i] = op(a[gid], val);
  }
}

template <typename Op>
__global__ void EwiseUnaryOpKernel(const scalar_t *a, scalar_t *out,
                                   size_t size, Op op) {
  // Calculate the global index of the thread — assuming we called CudaOneDim()
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;

  for (size_t i = gid; i < size; i += stride) {
    out[i] = op(a[gid]);
  }
}

template <typename Op>
void EwiseBinaryOp(const CudaArray &a, const CudaArray &b, CudaArray *out,
                   Op op) {
  CudaDims dim = CudaOneDim(out->size);
  EwiseBinaryOpKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr,
                                               out->size, op);
}

template <typename Op>
void ScalarBinaryOp(const CudaArray &a, const scalar_t val, CudaArray *out,
                    Op op) {
  CudaDims dim = CudaOneDim(out->size);
  ScalarBinaryOpKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size,
                                                op);
}

template <typename Op>
void EwiseUnaryOp(const CudaArray &a, CudaArray *out, Op op) {
  CudaDims dim = CudaOneDim(out->size);
  EwiseUnaryOpKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size, op);
}
/// END TEMPLATES

/// BEGIN DEVICE FUNCTORS
struct DeviceAdd {
  __device__ inline float operator()(float x, float y) { return x + y; }
};

struct DeviceMul {
  __device__ inline float operator()(float x, float y) { return x * y; }
};

struct DeviceDiv {
  __device__ inline float operator()(float x, float y) { return x / y; }
};

struct DevicePower {
  __device__ inline float operator()(float x, float y) {
    return std::pow(x, y);
  }
};

struct DeviceMax {
  __device__ inline float operator()(float x, float y) { return max(x, y); }
};

struct DeviceEq {
  __device__ inline float operator()(float x, float y) { return x == y; }
};

struct DeviceGe {
  __device__ inline float operator()(float x, float y) { return x >= y; }
};

struct DeviceLog {
  __device__ inline float operator()(float x) { return std::log(x); }
};

struct DeviceExp {
  __device__ inline float operator()(float x) { return std::exp(x); }
};

struct DeviceTanh {
  __device__ inline float operator()(float x) { return std::tanh(x); }
};

/// END DEVICE FUNCTORS

/// BEGIN SOLUTION
void EwiseMul(const CudaArray &a, const CudaArray &b, CudaArray *out) {
  EwiseBinaryOp(a, b, out, DeviceMul());
}

void ScalarMul(const CudaArray &a, scalar_t val, CudaArray *out) {
  ScalarBinaryOp(a, val, out, DeviceMul());
}

void EwiseDiv(const CudaArray &a, const CudaArray &b, CudaArray *out) {
  EwiseBinaryOp(a, b, out, DeviceDiv());
}

void ScalarDiv(const CudaArray &a, scalar_t val, CudaArray *out) {
  ScalarBinaryOp(a, val, out, DeviceDiv());
}

void ScalarPower(const CudaArray &a, scalar_t val, CudaArray *out) {
  ScalarBinaryOp(a, val, out, DevicePower());
}

void EwiseMaximum(const CudaArray &a, const CudaArray &b, CudaArray *out) {
  EwiseBinaryOp(a, b, out, DeviceMax());
}

void ScalarMaximum(const CudaArray &a, scalar_t val, CudaArray *out) {
  ScalarBinaryOp(a, val, out, DeviceMax());
}

void EwiseEq(const CudaArray &a, const CudaArray &b, CudaArray *out) {
  EwiseBinaryOp(a, b, out, DeviceEq());
}

void ScalarEq(const CudaArray &a, scalar_t val, CudaArray *out) {
  ScalarBinaryOp(a, val, out, DeviceEq());
}

void EwiseGe(const CudaArray &a, const CudaArray &b, CudaArray *out) {
  EwiseBinaryOp(a, b, out, DeviceGe());
}

void ScalarGe(const CudaArray &a, scalar_t val, CudaArray *out) {
  ScalarBinaryOp(a, val, out, DeviceGe());
}

void EwiseLog(const CudaArray &a, CudaArray *out) {
  EwiseUnaryOp(a, out, DeviceLog());
}

void EwiseExp(const CudaArray &a, CudaArray *out) {
  EwiseUnaryOp(a, out, DeviceExp());
}

void EwiseTanh(const CudaArray &a, CudaArray *out) {
  EwiseUnaryOp(a, out, DeviceTanh());
}
/// END SOLUTION

////////////////////////////////////////////////////////////////////////////////
// Elementwise and scalar operations
////////////////////////////////////////////////////////////////////////////////

/**
 * @todo Check for out-of-bounds errors before reading/writing!
 */
__global__ void MatmulKernel(const scalar_t *A, // M x N —— A.T is N x M
                             const scalar_t *B, // N x P
                             scalar_t *out,     // M x P
                             uint32_t M, uint32_t N, uint32_t P) {
  __shared__ scalar_t sA[S][L], sB[S][L];
  scalar_t c[V][V] = {0};
  scalar_t a[V], b[V];
  int row, col;
  int xblock = blockIdx.x;
  int yblock = blockIdx.y;

  for (int ko = 0; ko < N; ko += S) {
    __syncthreads();
    // Cooperative fetching to shared memories sA, sB
    int nthreads = blockDim.y * blockDim.x;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    for (int j = 0; j < S * L / nthreads; j++) {
      int cell_id = j * nthreads + tid;
      int y = cell_id / L;
      int x = cell_id % L;

      row = ko + y;

      // copy into A
      col = yblock * L + x;
      if (col < M && row < N) {
        sA[y][x] = A[col * N + row]; // A.T[row,col] = A[col,row]
      } else {
        sA[y][x] = 0;
      }

      // copy into B
      col = xblock * L + x;
      if (row < N && col < P) {
        sB[y][x] = B[row * P + col];
      } else {
        sB[y][x] = 0;
      }
    }
    __syncthreads();

    for (int ki = 0; ki < S; ki++) {
      // initialize each thread's a, b <-- sA, sB
      for (int Vi = 0; Vi < V; Vi++) {
        a[Vi] = sA[ki][threadIdx.y * V + Vi];
        b[Vi] = sB[ki][threadIdx.x * V + Vi];
      }
      //
      for (int y = 0; y < V; y++) {
        for (int x = 0; x < V; x++) {
          c[y][x] += a[y] * b[x];
        }
      }
    }
  }
  // thread: final V x V results to output array
  int ybase = blockIdx.y * blockDim.y + threadIdx.y;
  int xbase = blockIdx.x * blockDim.x + threadIdx.x;
  for (int y = 0; y < V; y++) {
    for (int x = 0; x < V; x++) {
      row = ybase * V + y;
      col = xbase * V + x;
      if (row < M && col < P) {
        out[row * P + col] = c[y][x];
      }
    }
  }
}

void Matmul(const CudaArray &a, const CudaArray &b, CudaArray *out, uint32_t M,
            uint32_t N, uint32_t P) {
  /**
   * Multiply two (compact) matrices into an output (also comapct) matrix.
   *
   * You will want to look at the lecture and notes on GPU-based linear algebra
   * to see how to do this. Since ultimately mugrade is just evaluating
   * correctness, you _can_ implement a version that simply parallelizes
   * over (i,j) entries in the output array. However, to really get the full
   * benefit of this problem, we would encourage you to use cooperative
   * fetching, shared memory register tiling, and other ideas covered in the
   * class notes.
   *
   * Note that unlike the tiled matmul function in the CPU backend, here you
   * should implement a single function that works across all size matrices,
   * whether or not they are a multiple of a tile size.
   *
   * As with previous CUDA implementations, this function here will largely just
   * set up the kernel call, and you should implement the logic in a separate
   * MatmulKernel() call.
   *
   * Args:
   *   a: compact 2D array of size m x n
   *   b: comapct 2D array of size n x p
   *   out: compact 2D array of size m x p to write the output to
   *   M: rows of a / out
   *   N: columns of a / rows of b
   *   P: columns of b / out
   */

  /// BEGIN SOLUTION
  // Compute number of blocks, and block dimensions (number of threads per
  // block)
  CudaDims dim;
  dim.block = dim3(16, 16, 1);
  dim.grid = dim3((P + L - 1) / L, (M + L - 1) / L, 1);

  MatmulKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, M, N, P);
  /// END SOLUTION
}

__global__ void MatmulNaiveKernel(const scalar_t *a, const scalar_t *b,
                                  scalar_t *out, uint32_t M, uint32_t N,
                                  uint32_t P) {
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= M || j >= P) {
    return;
  }

  scalar_t c = 0.0;
  for (int k = 0; k < N; k++) {
    c += a[i * N + k] * b[k * P + j];
  }
  out[i * P + j] = c;
}

void MatmulNaive(const CudaArray &a, const CudaArray &b, CudaArray *out,
                 uint32_t M, uint32_t N, uint32_t P) {
  /// BEGIN SOLUTION
  // Compute number of blocks, and block dimensions (number of threads per
  // block)
  CudaDims dim;
  dim.block = dim3(16, 16, 1);
  dim.grid = dim3((P + 15) / 16, (M + 15) / 16, 1);

  MatmulNaiveKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, M, N, P);
  /// END SOLUTION
}

////////////////////////////////////////////////////////////////////////////////
// Max and sum reductions
////////////////////////////////////////////////////////////////////////////////

template <typename Op>
__global__ void ReduceOpKernel(const scalar_t *a, scalar_t *out,
                               const size_t size, const size_t reduce_size,
                               Op op) {
  // Initialize gid, stride
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = gridDim.x * blockDim.x;

  // Write to out[i]
  for (size_t i = gid; i < size; i += stride) {
    // Loop over corresponding elements in a
    scalar_t acc = a[gid * reduce_size];
    for (size_t j = 1; j < reduce_size; ++j) {
      acc = op(acc, a[gid * reduce_size + j]);
    }
    out[i] = acc;
  }
}

template <typename Op>
void ReduceOp(const CudaArray &a, CudaArray *out, size_t reduce_size, Op op) {
  CudaDims dim = CudaOneDim(out->size);
  ReduceOpKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size,
                                          reduce_size, op);
}

void ReduceMax(const CudaArray &a, CudaArray *out, size_t reduce_size) {
  /**
   * Reduce by taking maximum over `reduce_size` contiguous blocks.  Even though
   * it is inefficient, for simplicity you can perform each reduction in a
   * single CUDA thread.
   *
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   reduce_size: size of the dimension to reduce over
   */
  /// BEGIN SOLUTION
  ReduceOp(a, out, reduce_size, DeviceMax());
  /// END SOLUTION
}

void ReduceSum(const CudaArray &a, CudaArray *out, size_t reduce_size) {
  /**
   * Reduce by taking summation over `reduce_size` contiguous blocks.  Again,
   * for simplicity you can perform each reduction in a single CUDA thread.
   *
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   reduce_size: size of the dimension to reduce over
   */
  /// BEGIN SOLUTION
  ReduceOp(a, out, reduce_size, DeviceAdd());
  /// END SOLUTION
}

} // namespace cuda
} // namespace needle

PYBIND11_MODULE(ndarray_backend_cuda, m) {
  namespace py = pybind11;
  using namespace needle;
  using namespace cuda;

  m.attr("__device_name__") = "cuda";
  m.attr("__tile_size__") = TILE;

  py::class_<CudaArray>(m, "Array")
      .def(py::init<size_t>(), py::return_value_policy::take_ownership)
      .def_readonly("size", &CudaArray::size)
      .def("ptr", &CudaArray::ptr_as_int);

  // return numpy array, copying from CPU
  m.def("to_numpy", [](const CudaArray &a, std::vector<size_t> shape,
                       std::vector<size_t> strides, size_t offset) {
    std::vector<size_t> numpy_strides = strides;
    std::transform(numpy_strides.begin(), numpy_strides.end(),
                   numpy_strides.begin(),
                   [](size_t &c) { return c * ELEM_SIZE; });

    // copy memory to host
    scalar_t *host_ptr = (scalar_t *)std::malloc(a.size * ELEM_SIZE);
    if (host_ptr == 0)
      throw std::bad_alloc();
    cudaError_t err =
        cudaMemcpy(host_ptr, a.ptr, a.size * ELEM_SIZE, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
      throw std::runtime_error(cudaGetErrorString(err));

    // return numpy array
    py::capsule deallocate_buffer(host_ptr, [](void *p) { free(p); });
    return py::array_t<scalar_t>(shape, numpy_strides, host_ptr + offset,
                                 deallocate_buffer);
  });

  // copy numpy array to GPU
  m.def("from_numpy", [](py::array_t<scalar_t> a, CudaArray *out) {
    cudaError_t err = cudaMemcpy(out->ptr, a.request().ptr,
                                 out->size * ELEM_SIZE, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
      throw std::runtime_error(cudaGetErrorString(err));
  });

  m.def("fill", Fill);
  m.def("compact", Compact);
  m.def("ewise_setitem", EwiseSetitem);
  m.def("scalar_setitem", ScalarSetitem);
  m.def("ewise_add", EwiseAdd);
  m.def("scalar_add", ScalarAdd);

  m.def("ewise_mul", EwiseMul);
  m.def("scalar_mul", ScalarMul);
  m.def("ewise_div", EwiseDiv);
  m.def("scalar_div", ScalarDiv);
  m.def("scalar_power", ScalarPower);

  m.def("ewise_maximum", EwiseMaximum);
  m.def("scalar_maximum", ScalarMaximum);
  m.def("ewise_eq", EwiseEq);
  m.def("scalar_eq", ScalarEq);
  m.def("ewise_ge", EwiseGe);
  m.def("scalar_ge", ScalarGe);

  m.def("ewise_log", EwiseLog);
  m.def("ewise_exp", EwiseExp);
  m.def("ewise_tanh", EwiseTanh);

  m.def("matmul", Matmul);
  m.def("matmul_naive", MatmulNaive);

  m.def("reduce_max", ReduceMax);
  m.def("reduce_sum", ReduceSum);
}
