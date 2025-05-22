"""
**Submitted to ANAC 2024 SCML (OneShot track)**
*Team* type your team name here
*Authors* type your team member names with their emails here

This code is free to use or update given that proper attribution is given to
the authors and the ANAC 2024 SCML competition.
"""

from scml.oneshot.rl.agent import OneShotRLAgent
from scml.oneshot.rl.action import FlexibleActionManager
from scml.oneshot.rl.common import model_wrapper

from .common import MODEL_PATH, MyObservationManager, TrainingAlgorithm, make_context

from .mycontext import MySupplierContext, MyConsumerContext

from scml_agents.scml2024.oneshot.team_193.matching_pennies2 import MatchingPennies
from scml_agents.scml2024.oneshot.team_171.dist_redist_agent import DistRedistAgent
from scml_agents.scml2024.oneshot.teamyuzuru.epsilon_greedy_agent import EpsilonGreedyAgent
from scml_agents.scml2024.oneshot.team_abc.suzuka_agent import SuzukaAgent
# used to repeat the response to every negotiator.
from scml_agents.scml2024.oneshot.team_miyajima_oneshot.cautious import CautiousOneShotAgent
from scml.oneshot.agents.rand import (
    EqualDistOneShotAgent,
    NiceAgent,
    RandDistOneShotAgent,
)

class MyAgent(OneShotRLAgent):
    """
    This is the only class you *need* to implement. The current skeleton simply loads a single model
    that is supposed to be saved in MODEL_PATH (train.py can be used to train such a model).
    """

    def __init__(self, *args, **kwargs):
        # get full path to models (supplier and consumer models).
        base_name = MODEL_PATH.name
        self.paths = [
            MODEL_PATH.parent / f"{base_name}_supplier_4-8_PPO_tanh_double_400000",
            MODEL_PATH.parent / f"{base_name}_consumer_4-8_PPO_tanh_double_300000",
        ]
        models = tuple(model_wrapper(TrainingAlgorithm.load(_)) for _ in self.paths)
        # contexts = (make_context(as_supplier=True), make_context(as_supplier=False))
        contexts = (MySupplierContext(), MyConsumerContext())
        # update keyword arguments
        kwargs.update(
            dict(
                # load models from MODEL_PATH
                models=models,
                fallback_type=CautiousOneShotAgent,
                # create corresponding observation managers
                observation_managers=(
                    MyObservationManager(context=contexts[0]),
                    MyObservationManager(context=contexts[1]),
                ),
                action_managers=(
                    FlexibleActionManager(context=contexts[0]),
                    FlexibleActionManager(context=contexts[1]),
                ),
            )
        )
        # Initialize the base OneShotRLAgent with model paths and observation managers.
        super().__init__(*args, **kwargs)

class EpsilonGreedyAgent1(EpsilonGreedyAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
class EpsilonGreedyAgent2(EpsilonGreedyAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class EpsilonGreedyAgent3(EpsilonGreedyAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class MatchingPennies1(MatchingPennies):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class MatchingPennies2(MatchingPennies):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class MatchingPennies3(MatchingPennies):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
class DistRedistAgent1(DistRedistAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class DistRedistAgent2(DistRedistAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
class DistRedistAgent3(DistRedistAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


if __name__ == "__main__":
    from .helpers.runner import run

    run([MyAgent, 
         MatchingPennies,
        #  MatchingPennies1,
        #  MatchingPennies2,
        #  MatchingPennies3,
         DistRedistAgent,
        #  DistRedistAgent1,
        #  DistRedistAgent2,
        #  DistRedistAgent3,
         EpsilonGreedyAgent,
        #  EpsilonGreedyAgent1,
        #  EpsilonGreedyAgent2,
        #  EpsilonGreedyAgent3,
         SuzukaAgent,
        #  RandDistOneShotAgent,
        #  EqualDistOneShotAgent,
         ])
