from environments.BaseAviary import BaseAviary
from environments.BaseRLAviary import BaseRLAviary
from environments.ejc_cl_stage1 import EjcCLStage1
from environments.ejc_cl_stage2 import EjcCLStage2
from environments.ejc_cl_stage3 import EjcCLStage3
from environments.journal_stage1_Euler import JournalStage1Euler
from environments.journal_stage1_Euler_NoiseFree import JournalStage1EulerNoiseFree
from environments.journal_stage2_Euler import JournalStage2Euler
from environments.journal_stage3_Euler import JournalStage3Euler
from environments.CL_Stage1_S2R_e2e_dr import CLStage1Sim2RealDomainRandomization
from environments.CL_Stage2_S2R_e2e_dr import CLStage2Sim2RealDomainRandomization
from environments.CL_Stage3_S2R_e2e_dr import CLStage3Sim2RealDomainRandomization
from environments.CL_Stage1_S2R_e2e_dr_quat import CLStage1S2RE2EDRQuat
from environments.CL_Stage2_S2R_e2e_dr_quad import CLStage2S2RE2EDRQuat
from environments.CL_Stage3_S2R_e2e_dr_quad import CLStage3S2RE2EDRQuat
from environments.CL_Stage1_S2R_e2e_dr_new_reward import CLStage1S2RE2EDRNewReward
from environments.CL_Stage2_S2R_e2e_dr_new_reward import CLStage2S2RE2EDRNewReward

environment_map = {
    'EjcCLStage1': EjcCLStage1,
    'EjcCLStage2': EjcCLStage2,
    'EjcCLStage3': EjcCLStage3,
    'JournalStage1Euler': JournalStage1Euler,
    'JournalStage1EulerNoiseFree': JournalStage1EulerNoiseFree,
    'JournalStage2Euler': JournalStage2Euler,
    'JournalStage3Euler': JournalStage3Euler,
    'CLStage1Sim2RealDomainRandomization': CLStage1Sim2RealDomainRandomization,
    'CLStage2Sim2RealDomainRandomization': CLStage2Sim2RealDomainRandomization,
    'CLStage3Sim2RealDomainRandomization': CLStage3Sim2RealDomainRandomization,
    'CLStage1S2RE2EDRQuat': CLStage1S2RE2EDRQuat,
    'CLStage2S2RE2EDRQuat': CLStage2S2RE2EDRQuat,
    'CLStage3S2RE2EDRQuat': CLStage3S2RE2EDRQuat,
    'CLStage1S2RE2EDRNewReward': CLStage1S2RE2EDRNewReward,
    'CLStage2S2RE2EDRNewReward': CLStage2S2RE2EDRNewReward,
}