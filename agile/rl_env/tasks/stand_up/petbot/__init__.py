"""PetBot recovery registration. No upstream humanoid config is inherited."""

import gymnasium as gym

gym.register(
    id="PetBot-Recovery-AGILE-v0",
    entry_point="agile.petbot.mdp:PetBotRecoveryEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "agile.petbot.env_cfg:PetBotRecoveryEnvCfg",
        "rsl_rl_cfg_entry_point": "agile.petbot.agents:PetBotPpoRunnerCfg",
        "pre_learn_entry_point": "agile.petbot.pre_learn:pre_learn",
    },
)


for suffix, config in (
    ("A", "PetBotSweep2EnvCfg"),
    ("B", "PetBotSweep2HistoryEnvCfg"),
    ("C", "PetBotSweep2CurriculumEnvCfg"),
):
    gym.register(
        id=f"PetBot-Recovery-Sweep2-{suffix}-v0",
        entry_point="agile.petbot.mdp:PetBotRecoveryEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"agile.petbot.env_cfg:{config}",
            "rsl_rl_cfg_entry_point": "agile.petbot.agents:PetBotSweep2RunnerCfg",
            "pre_learn_entry_point": "agile.petbot.pre_learn:pre_learn",
        },
    )
