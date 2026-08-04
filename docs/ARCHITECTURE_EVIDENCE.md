# Refactoring evidence

This page maps the refactoring contracts to source boundaries and executable
regressions. The links are intentionally repository-relative so they remain
useful from local checkouts and code review.

| Contract | Implementation | Evidence |
| --- | --- | --- |
| Neutral architecture resolution stages | [`architecture.hpp`](../include/celeg/model/architecture.hpp), [`architecture.cpp`](../src/model/architecture.cpp) | [`architecture_resolution_test.cpp`](../tests/architecture_resolution_test.cpp) |
| Per-layer FFN dimensions | [`weight_plan.cpp`](../src/model/weight_plan.cpp) | [`weight_plan_test.cpp`](../tests/weight_plan_test.cpp) |
| Explicit token/numerical policy | [`resolved.hpp`](../include/celeg/model/resolved.hpp), [`resolved.cpp`](../src/model/resolved.cpp) | [`policy_test.cpp`](../tests/policy_test.cpp) |
| Packed workspace and metadata ownership | [`packed_workspace.hpp`](../include/celeg/backend/cuda/packed_workspace.hpp), [`packed_metadata.hpp`](../include/celeg/backend/cuda/packed_metadata.hpp) | [`packed_workspace_test.cpp`](../tests/packed_workspace_test.cpp) |
| Host commit boundary | [`packed_commit.hpp`](../include/celeg/backend/cuda/packed_commit.hpp), [`packed_commit.cpp`](../src/backend/cuda/model/packed_commit.cpp) | [`packed_commit_test.cpp`](../tests/packed_commit_test.cpp) |
| Zero-allocation packed steady state | [`resources.cpp`](../src/backend/cuda/model/resources.cpp), [`paged_prefill.cpp`](../src/backend/cuda/model/paged_prefill.cpp) | [`cuda_granite_test.cu`](../tests/cuda_granite_test.cu) |
| CUDA linear binding and isolated GEMM collaborators | [`gemm_dispatcher.hpp`](../include/celeg/backend/cuda/gemm_dispatcher.hpp), [`gemm_dispatcher.cu`](../src/backend/cuda/runtime/gemm_dispatcher.cu) | [`cuda_gguf_kernels_test.cu`](../tests/cuda_gguf_kernels_test.cu) |
| Checkpoint/backend boundary | [`weight_repository.hpp`](../include/celeg/checkpoint/weight_repository.hpp) | [`fake_repository_backend_boundary_test.cpp`](../tests/fake_repository_backend_boundary_test.cpp) |

Focused verification currently used for these boundaries:

```text
cmake --build out/windows-cuda-relwithdebinfo --target celeg_base celeg_cuda_backend --config RelWithDebInfo --parallel 4
python scripts/check_architecture_boundaries.py --root .
```

The CUDA test target requires a usable CUDA device; a successful compilation
is still recorded separately from runtime execution when the environment does
not provide one.

On 2026-08-04, the configured RelWithDebInfo tree completed all 67 CTest
tests, including the concurrent packed CUDA Granite regression.
