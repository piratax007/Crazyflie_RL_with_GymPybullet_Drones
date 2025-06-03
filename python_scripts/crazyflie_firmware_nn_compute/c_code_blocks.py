headers_nn_compute = """
#include "nn_compute.h"

#define g 9.82
#define mass 0.027
#define kf 3.16e-10
#define hoverRPM sqrtf((g * mass) / (4 * kf))

"""

output_arrays = """
static float output_0[64];
static float output_1[64];
static float output_2[4];
"""


forward_pass_function = """
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
        output_1[i] += mlp_extractor_policy_net_2_bias[1];
        output_1[i] = tanhf(output_1[i]);
    }
    
    for (int i = 0; i < structure[2][0]; i++) {
        output_2[i] = 0;
        for (int j = 0; j < structure[2][1]; j++) {
            output_2[i] += output_1[j] * action_net_weight[i][j];
        }
        output_2[i] += action_net_bias[i];
    }
    
    control_n->rpm_0 = hoverRPM * (1.0f + 0.05f * output_2[0]);
    control_n->rpm_1 = hoverRPM * (1.0f + 0.05f * output_2[1]);
    control_n->rpm_2 = hoverRPM * (1.0f + 0.05f * output_2[2]);
    control_n->rpm_3 = hoverRPM * (1.0f + 0.05f * output_2[3]);
}
"""