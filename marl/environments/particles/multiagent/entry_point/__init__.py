from gym.envs.registration import register

# Multiagent envs
# original versions (does not work as normal gym because the scenario is missing)
# ----------------------------------------

register(
    id='MultiagentSimple-v0',
    # entry_point='marl.environments.particles.multiagent.envs:SimpleEnv',
    entry_point='marl.environments.particles.multiagent.entry_point.environment:MultiAgentEnv',

    # FIXME(cathywu) currently has to be exactly max_path_length parameters in
    # rllab run script
    max_episode_steps=100,
)


register(
    id='MultiagentSimpleSpeakerListener-v0',
    entry_point='marl.environments.particles.multiagent.envs.entry_point:SimpleSpeakerListenerEnv',
    max_episode_steps=100,
)
# ------------------------------------


register(
    id='MultiagentCustom-v0',
    entry_point='marl.environments.particles.multiagent.entry_point.environment:MultiAgentEnvGym',
    # the scenario.setup is called with these kwargs, but should be possible to call once again with different values
    kwargs={'scenario': 'custom/custom_no_comm', 'num_agents': 2, 'num_landmarks': 3}
)
