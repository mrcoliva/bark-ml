from bark.world.evaluation import \
  EvaluatorGoalReached, EvaluatorCollisionEgoAgent, \
  EvaluatorCollisionDrivingCorridor, EvaluatorStepCount
from modules.runtime.commons.parameters import ParameterServer
from bark.geometry import *

from src.evaluators.evaluator import StateEvaluator

class GoalReached(StateEvaluator):
  def __init__(self,
               params=ParameterServer(),
               eval_agent=None):
    StateEvaluator.__init__(self, params)
    self._goal_reward = \
      self._params["ML"]["Evaluator"]["goal_reward",
        "Reward for reaching the goal.",
        1.]
    self._collision_penalty = \
      self._params["ML"]["Evaluator"]["collision_penalty",
        "Reward given for a collisions.",
        -100.]
    self._max_steps = \
      self._params["ML"]["Evaluator"]["max_steps",
        "Maximum steps per episode.",
        50]
    self._eval_agent = eval_agent

  def _add_evaluators(self):
    self._evaluators["goal_reached"] = EvaluatorGoalReached(self._eval_agent)
    self._evaluators["ego_collision"] = \
      EvaluatorCollisionEgoAgent(self._eval_agent)
    # TODO(@hart): Fix
    #self._evaluators["collision_driving_corridor"] = \
    #  EvaluatorCollisionDrivingCorridor()
    self._evaluators["step_count"] = EvaluatorStepCount()

  
  def _distance_to_center_line(self, world):
    agent = world.agents[self._eval_agent]
    agent_state = agent.state
    centerline = agent.local_map.get_driving_corridor().center
    agent_xy = Point2d(agent.state[1], agent.state[2])
    return distance(centerline, agent_xy)

  def _evaluate(self, world, eval_results):
    """Returns information about the current world state
    """
    done = False
    success = eval_results["goal_reached"]
    distance = self._distance_to_center_line(world)
    # TODO(@hart): Fix
    collision = eval_results["ego_collision"]# or \
    #eval_results["collision_driving_corridor"]
    step_count = eval_results["step_count"]
    # determine whether the simulation should terminate
    if success or collision or step_count > self._max_steps:
      done = True
    # calculate reward
    reward = collision * self._collision_penalty + \
      success * self._goal_reward  - 0.1*distance
    return reward, done, eval_results
    

