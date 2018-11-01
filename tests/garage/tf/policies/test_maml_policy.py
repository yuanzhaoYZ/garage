import tensorflow as tf

from garage.tf.envs import TfEnv
from garage.tf.policies import GaussianMLPPolicy
from garage.tf.policies.maml_policy import MamlPolicy
from tests.fixtures.envs.dummy import DummyBoxEnv, DummyDiscreteEnv


def test1():
    box_env = TfEnv(DummyBoxEnv())
    gaussian_mlp_policy = GaussianMLPPolicy(
            env_spec=box_env, hidden_sizes=(1, ))
    maml_policy = MamlPolicy(gaussian_mlp_policy, num_sampled_task=2)
    sess = tf.Session()
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("./graphh", sess.graph)


def test_assign():
    # Build graph
    a = tf.constant(1.5)
    b = tf.Variable(initial_value=2.)
    c = tf.Variable(initial_value=0.)
    # Update op
    # update_op = b.assign(b + 2.)
    # assign_op = tf.assign(c, b)
    loss = a * c
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    # sess.run(update_op)
    # sess.run(assign_op)
    import ipdb
    ipdb.set_trace()


def test_mini_maml():

    a = tf.Variable(initial_value=2.)
    b = tf.constant(1.5)
    init_graph = a * b
    all_vars = [a]
    operations = tf.get_default_graph().get_operations()
    # for v in all_vars:
        
    import ipdb
    ipdb.set_trace()


VARIABLE_STORE = set()

def test_overloaded_get_vars():
    from garage.tf.core.networks import mlp
    import mock

    in_dim = 2
    out_dim = 3
    input_var = tf.placeholder(shape=[None, in_dim], dtype=tf.float32)

    mlp1 = mlp(
        input_var=input_var,
        output_dim=out_dim,
        hidden_sizes=[64, 32],
        name="mlp1"
    )

    all_vars = tf.trainable_variables(scope="mlp1")

    # create gradient path from current params to one step adapted params
    vars_copied = {}
    with tf.name_scope("maml_adaptation"):
        for p in all_vars:
            grad = tf.placeholder(
                dtype=p.dtype, 
                shape=p.shape, 
                name="maml/grad/{}".format(p.name[:-2])
            )
            maml_p = p - grad
            vars_copied["maml/{}".format(p.name)] = maml_p

    # new get_variable function
    def _get_variable(name, shape=None, **kwargs):
        scope = tf.get_variable_scope()
        idx = 0
        fullname = "{}/{}:{}".format(scope.name, name, idx)
        while fullname in VARIABLE_STORE:
            idx += 1
            fullname = "{}/{}:{}".format(scope.name, name, idx)
        VARIABLE_STORE.add(fullname)
        return vars_copied[fullname]

    # overload the whole tf.get_variable function
    # this allows us to use an operation as a variable 
    from tensorflow.python.ops import variable_scope
    variable_scope.get_variable = _get_variable

    # recunstruct the same graph but with operations 
    with tf.variable_scope("maml"):
        mlp2 = mlp(
            input_var=input_var,
            output_dim=out_dim,
            hidden_sizes=[64, 32],
            name="mlp1"
        )
    sess = tf.Session()
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("./graphh", sess.graph)


if __name__ == '__main__':
    test_overloaded_get_vars()