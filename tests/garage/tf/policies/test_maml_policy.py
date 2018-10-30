"""
This script creates a test that fails when
garage.tf.policies failed to initialize.
"""
from garage.tf.envs import TfEnv
from garage.tf.policies import CategoricalGRUPolicy
from garage.tf.policies import CategoricalLSTMPolicy
from garage.tf.policies import CategoricalMLPPolicy
from garage.tf.policies import ContinuousMLPPolicy
from garage.tf.policies import DeterministicMLPPolicy
from garage.tf.policies import GaussianGRUPolicy
from garage.tf.policies import GaussianLSTMPolicy
from garage.tf.policies import GaussianMLPPolicy
from garage.tf.policies.maml_policy import MamlPolicy
from tests.fixtures import TfGraphTestCase
from tests.fixtures.envs.dummy import DummyBoxEnv, DummyDiscreteEnv


class TestTfPolicies(TfGraphTestCase):
    def test_policies(self):
        """Test the policies initialization."""
        box_env = TfEnv(DummyBoxEnv())
        discrete_env = TfEnv(DummyDiscreteEnv())
        categorical_gru_policy = MamlPolicy(CategoricalGRUPolicy(
            env_spec=discrete_env, hidden_dim=1), 0)
        categorical_lstm_policy = MamlPolicy(CategoricalLSTMPolicy(
            env_spec=discrete_env, hidden_dim=1), 0)
        categorical_mlp_policy = MamlPolicy(CategoricalMLPPolicy(
            env_spec=discrete_env, hidden_sizes=(1, )), 0)
        continuous_mlp_policy = MamlPolicy(ContinuousMLPPolicy(
            env_spec=box_env, hidden_sizes=(1, )), 0)
        deterministic_mlp_policy = MamlPolicy(DeterministicMLPPolicy(
            env_spec=box_env, hidden_sizes=(1, )), 0)
        gaussian_gru_policy = MamlPolicy(GaussianGRUPolicy(env_spec=box_env, hidden_dim=1), 0)
        gaussian_lstm_policy = MamlPolicy(GaussianLSTMPolicy(
            env_spec=box_env, hidden_dim=1), 0)
        gaussian_mlp_policy = MamlPolicy(GaussianMLPPolicy(
            env_spec=box_env, hidden_sizes=(1, )), 0)
