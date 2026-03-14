# Tasks for SOFIE Alpaka GPU Inference

## Exercise 3: Exploring the SOFIE Alpaka Architecture

### Proposed Improvements and Extensions

#### 1. Continue operators implementation for the remaining operators, especially for important ones (computation bounded) like Conv... 

#### 2. Support Kernel Fusion
BatchNorm in exercise 5 allows optional fusion with ReLU by a flag. We can possibly support for other common operators combinations as well.

#### 3. Performance comparison btw GPU and CPU
Currently, it is more focus on correctness in this task but I think it is also important to conduct experiments to know\predicts how much and when GPU accelerates in usage. 

### Other ideas that may worth discussion (lower feasibility?):

#### 4. CUDA Graph Support 
This is related to kernel fusion suggestion. If it is found that kernel launching may be a overhead for some tasks. It may be beneficial to support CUDA Graph. It maybe difficult to implement as the batch size or other sizes are dynamic. We can still pre-define several fixed sizes to match and execute with paddings like vLLM. But again this is very task dependent so the feasibility of this suggestion can be low.

#### 5. Pipelining/Asynchronous inference
Currently, `infer()` ends with `alpaka::wait(queue)`. For batched input, it maybe beneficial to allow some pipelining to further improve performance.

---

## Exercise 4 & 5: Implement New GPU Operators

### Summary

I added GPU inference support for three ONNX operators: Elu, Softmax, and BatchNormalization, with corresponding tests.

### Details

#### 1. Elu

- Implementation mainly follows from SOFIE_ROPERATOR_LeakyRelu
- Each thread handles one element
- Formula: `out[i] = x[i] >= 0 ? x[i] : alpha * (exp(x[i]) - 1)`
- Parameters passed at code-generation time: `alpha` (as a `constexpr float`)

#### 2. Softmax

Softmax is more complicated as it requires the max element in the axis and the normalization terms which both require coordination across threads.

- One block of 256 threads is assigned to each independent softmax computation. All threads in the block cooperate via shared memory to compute the reductions.
- Algorithm: Numerically stable 3-pass approach:
  1. Max-reduce: Each thread finds its local max, then a tree reduction in shared memory produces the global max. Block-strided loop is used to maxmize memory coalescing in a warp. Shared memory is used for warps communication, specifically finding global maximum in the block.
  2. Exp + sum-reduce: Each thread computes `exp(x_i - max)` for its elements, writes results to output, and accumulates a local sum. The reduction to find sum is similar as Max-reduce.
  3. Normalize: Each thread divides its assigned output elements by the global sum. There is a read and a write to global memory for each thread. However, the memory coalescing in this part depends on the innersize.
- Possible improvement to be done:
  - Allow dynamic block size selection for different axis size.
  - Can possibly use `__shfl_down_sync` for intra-warp reduction first.

#### 3. BatchNormalization

- Each thread handles one element
- The `Initialize()` method already does broadcasting the tensor and compute the scale. Thus, the GPU kernel only needs `Y[i] = (X[i] - mean[i]) * scale[i] + bias[i]`.
- ReLU fusion: The kernel accepts a compile-time `bool applyRelu` flag, enabling fused BatchNorm+ReLU without a separate kernel launch.

AI assistance was used to support codebase comprehension, debugging, and implementation tasks (Claude Opus 4.6).