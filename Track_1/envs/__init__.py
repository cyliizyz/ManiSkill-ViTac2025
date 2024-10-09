from gymnasium.envs.registration import make, register, registry, spec

register(
    id='ContinuousInsertionRandomizedMarkerEnv-v1',
    entry_point='envs.peg_insertion:ContinuousInsertionSimGymRandomizedPointFLowEnv',
)

register(
    id='LongOpenLockRandPointFlowEnv-v1',
    entry_point='envs.long_open_lock:LongOpenLockRandPointFlowEnv',
)
