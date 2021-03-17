from dopamine.agents.dqn import dqn_agent
from dopamine.discrete_domains import atari_lib
import tensorflow as tf
import tensorflow_probability as tfp

import gin.tf


@gin.configurable
class RFFDQNAgent(dqn_agent.DQNAgent):
  """An implementation of the DQN agent."""

  def __init__(
            self,
            sess,
            num_actions,
            scale=None,
            network=atari_lib.RFFDQNNetwork,
            **kwargs):

        self.scale = scale
        dqn_agent.DQNAgent.__init__(
            self,
            sess=sess,
            num_actions=num_actions,
            network=network,
            **kwargs)

  def _build_train_op(self):
    """Builds a training op.

    Returns:
      train_op: An op performing one step of training from replay data.
    """
    replay_action_one_hot = tf.one_hot(
        self._replay.actions, self.num_actions, 1., 0., name='action_one_hot')
    replay_chosen_q = tf.reduce_sum(
        self._replay_net_outputs.q_values * replay_action_one_hot,
        axis=1,
        name='replay_chosen_q')

    rffs = self._replay_net_outputs.rff
    inner_prods = tf.tensordot(rffs, tf.transpose(rffs), 1)
    diag = tf.linalg.tensor_diag_part(inner_prods)
    diag_mean = tf.math.reduce_mean(diag)
    normalized = inner_prods / diag_mean

    inner_prod_mean = tf.math.reduce_mean(normalized)
    inner_prod_variance = tfp.stats.variance(tf.reshape(normalized, [-1]))

    target = tf.stop_gradient(self._build_target_q_op())
    loss = tf.compat.v1.losses.huber_loss(
        target, replay_chosen_q, reduction=tf.losses.Reduction.NONE)
    if self.summary_writer is not None:
      with tf.compat.v1.variable_scope('Losses'):
        tf.compat.v1.summary.scalar('HuberLoss', tf.reduce_mean(loss))
      with tf.compat.v1.variable_scope('Scale'):
        tf.compat.v1.summary.scalar('InnerProdVariance', inner_prod_variance)
        tf.compat.v1.summary.scalar('InnerProdMean', inner_prod_mean)
    return self.optimizer.minimize(tf.reduce_mean(loss))

  def _create_network(self, name):
      network = self.network(self.num_actions, self.scale, name=name)
      return network
