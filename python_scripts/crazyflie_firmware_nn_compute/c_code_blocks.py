headers_nn_compute = {'for-based':"""
#include "nn_compute.h"

#define g 9.82f
#define mass 0.033f
#define kf 3.16e-10f
#define hoverRPM sqrtf((g * mass) / (4 * kf))

""",
'unrolled-for':"""
#include "nn_compute.h"

#define g 9.82f
#define mass 0.033f
#define kf 3.16e-10f

typedef enum { ACT_LINEAR = 0, ACT_TANH = 1 } nn_activation_t;
static const float kHoverRPM = sqrtf((g * mass) / (4.0f * kf));

"""}

output_arrays = """
static float output_0[64];
static float output_1[64];
static float output_2[4];
"""


forward_pass_function = {'for-based':"""
void neuralNetworkComputation(struct control_t_n *control_n, const float *state_array) {
    for (int i = 0; i < structure[0][0]; i++) {
        output_0[i] = 0;
        for (int j = 0; j < structure[0][1]; j++) {
            output_0[i] += state_array[j] * mlp_extractor_policy_net_0_weight[i][j];
        }
        output_0[i] += mlp_extractor_policy_net_0_bias[i];
        output_0[i] = tanhf(output_0[i]);
    }
    
    for (int i = 0; i < structure[1][0]; i++) {
        output_1[i] = 0;
        for (int j = 0; j < structure[1][1]; j++) {
            output_1[i] += output_0[j] * mlp_extractor_policy_net_2_weight[i][j];
        }
        output_1[i] += mlp_extractor_policy_net_2_bias[i];
        output_1[i] = tanhf(output_1[i]);
    }
    
    for (int i = 0; i < structure[2][0]; i++) {
        output_2[i] = 0;
        for (int j = 0; j < structure[2][1]; j++) {
            output_2[i] += output_1[j] * action_net_weight[i][j];
        }
        output_2[i] += action_net_bias[i];
    }
    
    for (int i = 0; i < 4; i++) {
        output_2[i] = output_2[i] < -1.0f ? -1.0f : (output_2[i] > 1.0f ? 1.0f : output_2[i]);
    }
    
    control_n->rpm_0 = hoverRPM * (1.0f + 0.05f * output_2[0]);
    control_n->rpm_1 = hoverRPM * (1.0f + 0.05f * output_2[1]);
    control_n->rpm_2 = hoverRPM * (1.0f + 0.05f * output_2[2]);
    control_n->rpm_3 = hoverRPM * (1.0f + 0.05f * output_2[3]);
}
""",
'unrolled-for':"""
static inline void dense_fma(
    const float * __restrict x,
    const float * __restrict W,
    const float * __restrict b,
    float * __restrict y,
    const int out_dim,
    const int in_dim,
    const nn_activation_t act
) {
    for (int i = 0; i < out_dim; i++) {
        const float * __restrict Wi = &W[(size_t)i * (size_t)in_dim];
        float acc = b[i];

    #pragma GCC unroll 16

        for (int j = 0; j < in_dim; j++) {
            acc = fmaf(x[j], Wi[j], acc);
        }

        y[i] = (act == ACT_TANH) ? tanhf(acc) : acc;
    }
}

void neuralNetworkComputation(struct control_t_n *control_n, const float *state_array) {
    dense_fma(
        state_array,
        &mlp_extractor_policy_net_0_weight[0][0],
        &mlp_extractor_policy_net_0_bias[0],
        &output_0[0],
        structure[0][0], structure[0][1], ACT_TANH
    );

    dense_fma(
        &output_0[0],
        &mlp_extractor_policy_net_2_weight[0][0],
        &mlp_extractor_policy_net_2_bias[0],
        &output_1[0],
        structure[1][0], structure[1][1], ACT_TANH
    );

    dense_fma(
        &output_1[0],
        &action_net_weight[0][0],
        &action_net_bias[0],
        &output_2[0],
        structure[2][0], structure[2][1], ACT_LINEAR
    );
    
    for (int i = 0; i < 4; i++) {
        output_2[i] = output_2[i] < -1.0f ? -1.0f : (output_2[i] > 1.0f ? 1.0f : output_2[i]);
    }
    
    control_n->rpm_0 = kHoverRPM * (1.0f + 0.05f * output_2[0]);
    control_n->rpm_1 = kHoverRPM * (1.0f + 0.05f * output_2[1]);
    control_n->rpm_2 = kHoverRPM * (1.0f + 0.05f * output_2[2]);
    control_n->rpm_3 = kHoverRPM * (1.0f + 0.05f * output_2[3]);
}

"""
}