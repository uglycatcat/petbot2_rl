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


# Independent action-distribution repair, without silently changing old experiment configurations.
gym.register(
    id="PetBot-Recovery-Repair-v0",
    entry_point="agile.petbot.mdp:PetBotRecoveryEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "agile.petbot.env_cfg:PetBotSweep2HistoryEnvCfg",
        "rsl_rl_cfg_entry_point": "agile.petbot.agents:PetBotRepairRunnerCfg",
        "pre_learn_entry_point": "agile.petbot.pre_learn:pre_learn",
    },
)


for suffix, config, runner in (
    ("A", "PetBotSweep2HistoryEnvCfg", "PetBotRepairRunnerCfg"),
    ("B", "PetBotSweep3EnvCfg", "PetBotSweep3RunnerCfg"),
    ("C", "PetBotSweep3CurriculumEnvCfg", "PetBotSweep3RunnerCfg"),
):
    gym.register(
        id=f"PetBot-Recovery-Sweep3-{suffix}-v0",
        entry_point="agile.petbot.mdp:PetBotRecoveryEnv", disable_env_checker=True,
        kwargs={"env_cfg_entry_point": f"agile.petbot.env_cfg:{config}",
                "rsl_rl_cfg_entry_point": f"agile.petbot.agents:{runner}",
                "pre_learn_entry_point": "agile.petbot.pre_learn:pre_learn"},
    )

for suffix, config in (("B", "PetBotSweep3EnvCfg"), ("C", "PetBotSweep3CurriculumEnvCfg")):
    gym.register(
        id=f"PetBot-Recovery-Sweep4-{suffix}-v0",
        entry_point="agile.petbot.mdp:PetBotRecoveryEnv", disable_env_checker=True,
        kwargs={"env_cfg_entry_point": f"agile.petbot.env_cfg:{config}",
                "rsl_rl_cfg_entry_point": "agile.petbot.agents:PetBotSweep4RunnerCfg",
                "pre_learn_entry_point": "agile.petbot.pre_learn:pre_learn"},
    )
