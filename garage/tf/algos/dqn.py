"""
This module implements a DQN model.
"""
import numpy as np
import tensorflow as tf

from garage.misc import logger
from garage.misc.overrides import overrides
from garage.tf.algos.off_policy_rl_algorithm import OffPolicyRLAlgorithm


class DQN(OffPolicyRLAlgorithm):
    """
    A DQN model based on https://arxiv.org/pdf/1509.02971.pdf.
    """

    def __init__(self,
                 env,
                 replay_buffer,
                 n_timestamps=int(1e5),
                 qf_lr=0.001,
                 smooth_return=False,
                 qf_optimizer=tf.train.AdamOptimizer,
                 discount=0.95,
                 name=None,
                 dueling=False,
                 target_network_update_freq=500,
                 **kwargs):
        """
        Construct class.
        """
        self.qf_lr = qf_lr
        self.qf_optimizer = qf_optimizer
        self.name = name
        self.n_timestamps = n_timestamps
        self.target_network_update_freq = target_network_update_freq
        self.dueling = dueling

        super(DQN, self).__init__(
            env=env,
            replay_buffer=replay_buffer,
            use_target=True,
            discount=discount,
            **kwargs)

    @overrides
    def init_opt(self):

        obs_dim = self.env.spec.observation_space.shape
        # assume discrete space for dqn, so always action_space.n
        action_dim = self.env.spec.action_space.n

        # build q networks
        with tf.name_scope(self.name, "DQN"):
            self.obs_t_ph = tf.placeholder(
                tf.float32, (None, ) + obs_dim, name="obs")
            self.action_t_ph = tf.placeholder(tf.int32, None, name="action")
            self.reward_t_ph = tf.placeholder(tf.float32, None, name="reward")
            self.done_t_ph = tf.placeholder(tf.float32, None, name="done")
            self.next_obs_t_ph = tf.placeholder(
                tf.float32, (None, ) + obs_dim, name="next_obs")

            self.q_val = self.qf._build_net(
                name="qf", input=self.obs_t_ph, dueling=self.dueling)

            if self.use_target:
                self.target_q_val = self.qf._build_net(
                    name="target_qf",
                    input=self.next_obs_t_ph,
                    dueling=self.dueling)

                self._qf_update_ops = get_target_ops(
                    self.qf.get_global_vars("qf"),
                    self.qf.get_global_vars("target_qf"))

            # Q-value of the selected action
            q_selected = tf.reduce_sum(
                self.q_val * tf.one_hot(self.action_t_ph, action_dim), 1)

            # Bellman RHS, calculated from target network
            if self.use_target:
                future_best_q_val = tf.reduce_max(self.target_q_val, 1)
            else:
                future_best_q_val = tf.reduce_max(self.q_val, 1)

            q_best_masked = (1.0 - self.done_t_ph) * future_best_q_val
            # if done, it's just reward
            # else reward + discount * future_best_q_val
            target_q_values = self.reward_t_ph + self.discount * q_best_masked

            # Create Ops
            # objective function: minimize error
            self._loss = tf.reduce_mean(
                tf.square(q_selected - tf.stop_gradient(target_q_values)))

            self._optimize_loss = self.qf_optimizer(self.qf_lr).minimize(
                self._loss, var_list=self.qf.get_trainable_vars())

    @overrides
    def train(self, sess=None):
        created_session = True if sess is None else False
        if sess is None:
            self.sess = tf.Session()
            self.sess.__enter__()
        else:
            self.sess = sess

        self.sess.run(tf.global_variables_initializer())
        self.start_worker(self.sess)

        if self.use_target:
            self.sess.run(self._qf_update_ops, feed_dict=dict())

        # Share the qf with the policy
        self.policy._init_qf(lambda x: self.sess.run(
            self.q_val, feed_dict={self.obs_t_ph: x}))

        episode_rewards = []
        episode_qf_losses = []
        last_average_return = None

        for itr in range(self.n_timestamps):
            with logger.prefix('iteration #%d | ' % itr):
                paths = self.obtain_samples(itr)
                samples_data = self.process_samples(itr, paths)
                episode_rewards.extend(samples_data["undiscounted_returns"])
                self.log_diagnostics(paths)

                if self.replay_buffer.n_transitions_stored >= self.min_buffer_size:  # noqa: E501
                    if itr % self.n_train_steps == 0:
                        self.evaluate = True
                        qf_loss = self.optimize_policy(itr, samples_data)
                        episode_qf_losses.append(qf_loss)
                    if itr % self.target_network_update_freq == 0:
                        self.sess.run(self._qf_update_ops, feed_dict=dict())

                if self.plot:
                    self.plotter.update_plot(self.policy, self.max_path_length)
                    if self.pause_for_plot:
                        input("Plotting evaluation run: Press Enter to "
                              "continue...")

                logger.log("Training finished")
                logger.log("Saving snapshot #{}".format(itr))
                params = self.get_itr_snapshot(itr, samples_data)
                logger.save_itr_params(itr, params)
                logger.log("Saved")

                if self.evaluate:
                    logger.record_tabular('Iteration', itr)
                    logger.record_tabular('AverageReturn',
                                          np.mean(episode_rewards))
                    logger.record_tabular('StdReturn', np.std(episode_rewards))
                    logger.record_tabular('QFunction/AverageQFunctionLoss',
                                          np.mean(episode_qf_losses))
                    last_average_return = np.mean(episode_rewards)

                if not self.smooth_return:
                    episode_rewards = []
                    episode_qf_losses = []

                logger.dump_tabular(with_prefix=False)

        self.shutdown_worker()
        if created_session:
            self.sess.close()
        return last_average_return

    @overrides
    def optimize_policy(self, itr, sample_data):
        transitions = self.replay_buffer.sample(self.buffer_batch_size)
        observations = transitions["observation"]
        rewards = transitions["reward"]
        actions = transitions["action"]
        next_observations = transitions["next_observation"]
        dones = transitions["terminal"]

        loss, _ = self.sess.run(
            [self._loss, self._optimize_loss],
            feed_dict={
                self.obs_t_ph: observations,
                self.action_t_ph: actions,
                self.reward_t_ph: rewards,
                self.done_t_ph: dones,
                self.next_obs_t_ph: next_observations
            })

        return loss

    @overrides
    def get_itr_snapshot(self, itr, samples_data):
        return dict(itr=itr, policy=self.policy, env=self.env)


def get_target_ops(variables, target_variables):
    """Get target network update operations."""
    init_ops = []
    assert len(variables) == len(target_variables)
    for var, target_var in zip(variables, target_variables):
        # hard update
        init_ops.append(tf.assign(target_var, var))

    return init_ops
