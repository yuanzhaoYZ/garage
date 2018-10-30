"""
Since tensorflow assumes static graph, the number of sampled task must be specified.

"""


import tensorflow as tf

# TODO: expose all policy api from the wrapped policy

class MamlPolicy:

    def __init__(self, wrapped_policy, num_sampled_task, name="MamlPolicy"):

        self._wrapped_policy = wrapped_policy
        self._num_sampled_task = num_sampled_task
        self.name = name

        self._parameters_storage = list()
        self._assign_ops = list()
        self._parameters = None

        self._init_maml_params()

    def _init_maml_params(self):
        for i in range(self._num_sampled_task):
            params_copy, copy_ops = self._make_params_copy(i)
            self._parameters_storage.append(params_copy)
            self._assign_ops.append(copy_ops)

        self._parameters, self._old_params_copy_op = self._make_params_copy("old")

        self._reset_to_old_params_op = list()
        for old_p, current_p in zip(self._parameters, self._wrapped_policy.get_params_internal()):
            self._reset_to_old_params_op.append(tf.assign(current_p, old_p))

    def _make_params_copy(self, copy_id):
        current_params = self._wrapped_policy.get_params_internal()

        params_copy = list()
        copy_ops = list()
        # TODO: add variable scope
        with tf.variable_scope(self.name, reuse=False):
            for p in current_params:
                # TODO: fix the names to the copied params
                p_copied = tf.get_variable(
                    name='{}/copied/{}'.format(p.name[:-2], copy_id),
                    shape=p.shape,
                    dtype=p.dtype,
                )
                params_copy.append(p_copied)
                copy_ops.append(tf.assign(p_copied, p))

        return params_copy, copy_ops

    def copy_params_value(self, target_index=None):
        sess = tf.get_default_session()
        if target_index is None:
            sess.run(self._old_params_copy_op)
        else:
            sess.run(self._assign_ops[target_index])

    def reset_to_old_params(self):
        sess = tf.get_default_session()
        sess.run(self._reset_to_old_params_op)

    #################### Below are the neccessary methods from wrapped policy for MAML #######################
    def get_action(self, observation):
        return self._wrapped_policy.get_action()

    def get_actions(self, observations):
        return self._wrapped_policy.get_actions() 

    def log_diagnostics(self, paths):
        self._wrapped_policy.log_diagnostics(paths)

    @property
    def distribution(self):
        return self._wrapped_policy.distribution
    