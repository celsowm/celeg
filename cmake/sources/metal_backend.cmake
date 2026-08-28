set(CELEG_METAL_BACKEND_SOURCES
    src/backend/metal/runtime/device.mm
    src/backend/metal/model/compiler.cpp
    src/backend/metal/model/resources.cpp
    src/backend/metal/model/lifecycle.cpp
)

set(CELEG_METAL_MODEL_SOURCES
    src/backend/metal/model/model.mm
)

set(CELEG_METAL_SHADER_SOURCES
    src/backend/metal/kernels/probe.metal
    src/backend/metal/kernels/inference.metal
)

set(CELEG_METAL_APP_SOURCES
    src/app/benchmark/metal/inference.cpp
)
