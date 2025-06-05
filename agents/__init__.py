from agents.crl_infonce import CRLInfoNCEAgent
from agents.dino_rebrac import DINOReBRACAgent
from agents.fb_repr import ForwardBackwardRepresentationAgent
from agents.hilp import HILPAgent
from agents.infom import InFOMAgent
from agents.iql import IQLAgent
from agents.mbpo_rebrac import MBPOReBRACAgent
from agents.rebrac import ReBRACAgent
from agents.td_infonce import TDInfoNCEAgent

agents = dict(
    crl_infonce=CRLInfoNCEAgent,
    dino_rebrac=DINOReBRACAgent,
    fb_repr=ForwardBackwardRepresentationAgent,
    hilp=HILPAgent,
    infom=InFOMAgent,
    iql=IQLAgent,
    mbpo_rebrac=MBPOReBRACAgent,
    rebrac=ReBRACAgent,
    td_infonce=TDInfoNCEAgent,
)
