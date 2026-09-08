/**
 * @brief matvec_input — split from vector.metal (full replace, Doxygen only).
 */
/**
 * @brief Output rows each simdgroup accumulates in one matvec pass.
 *
 * Holding several rows keeps the activation values in registers across them, so
 * the activation loads, the SwiGLU evaluation and the index arithmetic are paid
 * once per group of rows instead of once per row.
 */
constant uint kCelegMatvecRows = 4;

/// @brief Reads the matrix-vector input straight from a dense float vector.
struct CelegDenseInput {
    device const float* values;

    float at(uint column) const { return values[column]; }
};

/**
 * @brief Applies SwiGLU to a packed gate/up pair and feeds the result in.
 *
 * Fusing the activation into the down projection removes both the separate
 * elementwise dispatch and the round trip of the intermediate vector through
 * device memory.
 */
struct CelegSwigluInput {
    device const float* gate_up;
    uint cols;

    float at(uint column) const {
        const float gate = gate_up[column];
        return gate / (1.0f + exp(-gate)) * gate_up[cols + column];
    }
};
