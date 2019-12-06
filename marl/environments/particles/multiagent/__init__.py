from gym.envs.registration import register

# Multiagent envs
# original versions (does not work as normal gym because the scenario is missing)
# ----------------------------------------
"""
register(
    id='MultiagentSimple-v0',
    # entry_point='marl.environments.particles.multiagent.envs:SimpleEnv',
    entry_point='marl.environments.particles.multiagent.environment:MultiAgentEnv',

    # FIXME(cathywu) currently has to be exactly max_path_length parameters in
    # rllab run script
    max_episode_steps=100,
)


register(
    id='MultiagentSimpleSpeakerListener-v0',
    entry_point='marl.environments.particles.multiagent.envs:SimpleSpeakerListenerEnv',
    max_episode_steps=100,
)
# ------------------------------------


register(
    id='SimpleTest-v0',
    entry_point='marl.environments.particles.multiagent.environment:MultiAgentEnvGym',
    # max_episode_steps=100,
    # kwargs = {}
)
"""
