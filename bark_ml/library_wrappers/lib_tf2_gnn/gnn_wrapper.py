import time
import numpy as np
import tensorflow as tf
from tf2_gnn.layers import GNN, GNNInput
from bark.runtime.commons.parameters import ParameterServer
from bark_ml.observers.graph_observer import GraphObserver
from spektral.layers import GraphSageConv
from tensorflow.keras.layers import Dense
from spektral.layers import EdgeConditionedConv, GlobalSumPool, GraphSageConv, GraphAttention
from spektral.utils import numpy_to_batch


class GNNWrapper(tf.keras.Model):

  def __init__(self,
               params,
               name='GNN',
               **kwargs):
    super(GNNWrapper, self).__init__(name=name)

    mp_style = params["message_calculation_class", "", "rgcn"]
    gnn_params = GNN.get_default_hyperparameters(mp_style)
    
    if isinstance(params, ParameterServer):
      params = params.ConvertToDict()
    
    gnn_params.update(params)
    
    self._gnn = GNN(gnn_params)
    self.num_units = gnn_params["hidden_dim"]
    self.graph_conversion_times = []
    self.gnn_call_times = []

    self._conv_layers = [
      GraphAttention(channels=16),
      GraphAttention(channels=16)
    ]

    self._dense_layers = [
      Dense(self.num_units, activation='tanh')
    ]

  # def call(self, observations, training=False):
  #   t0 = time.time()
  #   features, edges = [], []
  #   node_to_graph_map = []
  #   num_graphs = tf.constant(len(observations))

  #   edge_index_offset = 0
  #   for i, sample in enumerate(observations):
  #     f, e = GraphObserver.gnn_input(sample)
  #     features.extend(f)
  #     num_nodes = len(f)
  #     e = np.asarray(e) + edge_index_offset
  #     edges.extend(e)
  #     node_to_graph_map.extend(np.full(num_nodes, i))
  #     edge_index_offset += num_nodes

  #   gnn_input = GNNInput(
  #     node_features=tf.convert_to_tensor(features),
  #     adjacency_lists=(
  #       tf.convert_to_tensor(edges, dtype=tf.int32),
  #     ),
  #     node_to_graph_map=tf.convert_to_tensor(node_to_graph_map),
  #     num_graphs=num_graphs,
  #   )

  #   self.graph_conversion_times.append(time.time() - t0)

  #   t0 = time.time()
  #   output = self._gnn(gnn_input, training=training)
  #   self.gnn_call_times.append(time.time() - t0)
  #   return output

  def call(self, observations, training=False):
    t0 = time.time()
    num_graphs = len(observations)

    X = []
    A = []
    E = np.ones(shape=(num_graphs, 0, 0, 0))
    
    for i, sample in enumerate(observations):
      n, a = GraphObserver.gnn_input(sample)
      X.append(np.array(n))
      A.append(np.array(a))

    X, A, E = numpy_to_batch(X, A, E)
    X = tf.convert_to_tensor(X, dtype=tf.float32)
    A = tf.convert_to_tensor(A, dtype=tf.float32)
    E = tf.convert_to_tensor(E, dtype=tf.float32)
    self.graph_conversion_times.append(time.time() - t0)

    t0 = time.time()
    for layer in self._conv_layers:
      X = layer([X, A])
    self.gnn_call_times.append(time.time() - t0)

    X = tf.reshape(X, [X.shape[0], -1])

    for layer in self._dense_layers:
      X = layer(X)
    
    return X

  def batch_call(self, graph, training=False):
    """Calls the network multiple times
    
    Arguments:
        graph {np.array} -- Graph representation
    
    Returns:
        np.array -- Batch of values
    """
    if graph.shape[0] == 0:
      return tf.random.normal(shape=(0, self.num_units))
    if len(graph.shape) > 3:
      raise ValueError(f'Graph has invalid shape {graph.shape}')

    return self.call(graph, training=training)
      
  def reset(self):
    self._gnn.reset()