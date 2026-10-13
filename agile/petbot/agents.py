"""PPO config using AGILE's RSL-RL 5 adapter; TensorBoard is the default logger."""

from isaaclab.utils import configclass

from agile.rl_env.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

from .collection import CollectionCfg


@configclass
class PetBotPpoRunnerCfg(RslRlOnPolicyRunnerCfg):
    seed = 42
    num_steps_per_env = 32
    max_iterations = 10000
    save_interval = 100
    experiment_name = "recovery_agile_01"
    run_name = "petbot"
    logger = "tensorboard"
    empirical_normalization = True
    clip_actions = 1.0
    # Initial-state generation has its own seed, independent of PPO seed.
    petbot_collection = CollectionCfg()
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.7,
        actor_hidden_dims=[128, 128, 128],
        critic_hidden_dims=[256, 128, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
