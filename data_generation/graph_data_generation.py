# Copyright (c) 2019 fortiss GmbH
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
import os
import networkx as nx
import matplotlib as mpl
"""if os.environ.get('DISPLAY') == ':0':
  print('No display found. Using non-interactive Agg backend')
  mpl.use('Agg')"""

from absl import app
from absl import flags

import time, json, pickle
import numpy as np
from abc import ABC
from tf_agents.environments import tf_py_environment

from src.observers.graph_observer import GraphObserver
from modules.runtime.commons.parameters import ParameterServer
from modules.runtime.scenario.scenario_generation.deterministic \
  import DeterministicScenarioGeneration
from modules.runtime.scenario.scenario_generation.configurable_scenario_generation \
  import ConfigurableScenarioGeneration
from src.wrappers.dynamic_model import DynamicModel
from src.evaluators.goal_reached import GoalReached
from modules.runtime.viewer.matplotlib_viewer import MPViewer
from src.rl_runtime import RuntimeRL

class DataGenerator(ABC):
  """Data Generator class"""
  def __init__(self, num_scenarios=10, step_time=0.1, steps=10, dump_dir=None):
    self.dump_dir = dump_dir
    self.steps = steps
    self.num_scenarios = num_scenarios
    ###############################################################################
    # Adapt params (start from default)
    params = ParameterServer()
    self.base_dir = os.path.dirname(os.path.dirname(__file__))
    params["BaseDir"] = self.base_dir
    params["Scenario"]["Generation"]["ConfigurableScenarioGeneration"]["MapFilename"] = "tests/data/city_highway_straight.xodr"
    self.params = params
    ###############################################################################
    # Start working with params
    self.scenario_generation = ConfigurableScenarioGeneration(num_scenarios = num_scenarios, params = self.params)

    self.observer = GraphObserver(self.params)
    self.behavior_model = DynamicModel(params=self.params)
    self.evaluator = GoalReached(params=self.params)
    self.viewer = MPViewer(params=self.params, use_world_bounds=True) # follow_agent_id=True)
  
    self.runtime = RuntimeRL(action_wrapper=self.behavior_model,
                              observer=self.observer,
                              evaluator=self.evaluator,
                              step_time=step_time,
                              viewer=self.viewer,
                              scenario_generator=self.scenario_generation)
    self.scenario_id = 0
    self.data = list()


  def run_scenario(self, scenario):
    """Runs a specific scenario from initial for predefined steps

        Inputs: scenario    :   bark-scenario
        
        Output: scenario_data : list containing all individual data_points of run scenario"""

    data_scenario = list()
    graph = self.runtime.reset(scenario)
    for i in range(self.steps):
        # Generate random steer and acc commands
        steer = np.random.random()*0.2 - 0.1
        acc = np.random.random()*1.0 - 1.0

        # Run step
        # [acc, steer] with -1<acc<1 and -0.1<steer<0.1
        (graph, actions), _, _, _ = self.runtime.step([acc, steer]) 
        # Save datum in data_scenario
        datum = dict()
        datum["graph"] = nx.node_link_data(graph)
        datum["actions"] = actions
        data_scenario.append(datum)
        self.runtime.render()
        time.sleep(0.01)

    return data_scenario

  def run_scenarios(self):
    """Run all scenarios"""
    
    for _ in range(0, self.num_scenarios):
      scenario, idx = self.scenario_generation.get_next_scenario()
      print("Scenario", idx)
      data_scenario = self.run_scenario(scenario)
      self.save_data(data_scenario)
      #self.data.append(data_scenario)
    
  def save_data(self, data):
      # Save data
      if self.dump_dir == None:
        raise Exception("specify dump_dir to tell the system where to store the data")
      else:
        if not os.path.exists(self.dump_dir):
          os.makedirs(self.dump_dir)
        path = self.dump_dir+ '/dataset_' + str(int(time.time())) + '.pickle'
      with open(path, 'wb') as handle:
          pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
          print('---> Dumped dataset to: ' + path)

def gen_data(argv):
  graph_generator = DataGenerator(dump_dir='/home/silvan/working_bark/data_generation/data/dummy_data')
  graph_generator.run_scenarios()


if __name__ == '__main__':
  app.run(gen_data)