from environments.BaseAviary import BaseAviary
from environments.BaseRLAviary import BaseRLAviary
from environments.CL_Stage1_S2R_e2e import CLStage1Sim2Real
from environments.CL_Stage2_S2R_e2e import CLStage2Sim2Real
from environments.CL_Stage3_S2R_e2e import CLStage3Sim2Real
from environments.CL_Stage1_S2R_e2e_dr_quat import CLStage1S2RE2EDRQuat
from environments.CL_Stage1_S2R_e2e_dr import CLStage1Sim2RealDomainRandomization
from environments.CL_Stage2_S2R_e2e_dr import CLStage2Sim2RealDomainRandomization
from environments.CL_Stage3_S2R_e2e_dr import CLStage3Sim2RealDomainRandomization

environment_map = {
    'CLStage1Sim2Real': CLStage1Sim2Real,
    'CLStage2Sim2Real': CLStage2Sim2Real,
    'CLStage3Sim2Real': CLStage3Sim2Real,
    'CLStage1S2RE2EDRQuat': CLStage1S2RE2EDRQuat,
    'CLStage1Sim2RealDomainRandomization': CLStage1Sim2RealDomainRandomization,
    'CLStage2Sim2RealDomainRandomization': CLStage2Sim2RealDomainRandomization,
    'CLStage3Sim2RealDomainRandomization': CLStage3Sim2RealDomainRandomization,
}