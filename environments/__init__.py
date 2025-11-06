from environments.BaseAviary import BaseAviary
from environments.BaseRLAviary import BaseRLAviary
from environments.safe_rl_simulation_stage1 import SafeRLSimulationStage1
from environments.safe_rl_simulation_stage2 import SafeRLSimulationStage2
from environments.safe_rl_simulation_stage3 import SafeRLSimulationStage3
from environments.CL_Stage1_S2R_e2e_dr import CLStage1Sim2RealDomainRandomization
from environments.CL_Stage2_S2R_e2e_dr import CLStage2Sim2RealDomainRandomization
from environments.CL_Stage3_S2R_e2e_dr import CLStage3Sim2RealDomainRandomization
from environments.CL_Stage1_S2R_e2e_dr_quat import CLStage1S2RE2EDRQuat
from environments.CL_Stage2_S2R_e2e_dr_quad import CLStage2S2RE2EDRQuat
from environments.CL_Stage3_S2R_e2e_dr_quad import CLStage3S2RE2EDRQuat
from environments.CL_Stage1_S2R_e2e_dr_new_reward import CLStage1S2RE2EDRNewReward
from environments.CL_Stage2_S2R_e2e_dr_new_reward import CLStage2S2RE2EDRNewReward

environment_map = {
    'SafeRLSimulationStage1': SafeRLSimulationStage1,
    'SafeRLSimulationStage2': SafeRLSimulationStage2,
    'SafeRLSimulationStage3': SafeRLSimulationStage3,
    'CLStage1Sim2RealDomainRandomization': CLStage1Sim2RealDomainRandomization,
    'CLStage2Sim2RealDomainRandomization': CLStage2Sim2RealDomainRandomization,
    'CLStage3Sim2RealDomainRandomization': CLStage3Sim2RealDomainRandomization,
    'CLStage1S2RE2EDRQuat': CLStage1S2RE2EDRQuat,
    'CLStage2S2RE2EDRQuat': CLStage2S2RE2EDRQuat,
    'CLStage3S2RE2EDRQuat': CLStage3S2RE2EDRQuat,
    'CLStage1S2RE2EDRNewReward': CLStage1S2RE2EDRNewReward,
    'CLStage2S2RE2EDRNewReward': CLStage2S2RE2EDRNewReward,
}