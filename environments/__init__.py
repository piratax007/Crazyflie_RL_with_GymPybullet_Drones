from environments.BaseAviary import BaseAviary
from environments.BaseRLAviary import BaseRLAviary
from environments.CL_Stage1_S2R_e2e import CLStage1Sim2Real
from environments.CL_Stage2_S2R_e2e import CLStage2Sim2Real
from environments.CL_Stage3_S2R_e2e import CLStage3Sim2Real
from environments.hovering import Hovering

environment_map = {
    'CLStage1Sim2Real': CLStage1Sim2Real,
    'CLStage2Sim2Real': CLStage2Sim2Real,
    'CLStage3Sim2Real': CLStage3Sim2Real,
    'Hovering': Hovering,
}