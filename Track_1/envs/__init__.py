from gymnasium.envs.registration import make, register, registry, spec

register(
    id="PegInsertionRandomizedMarkerEnv-v1",
    entry_point="envs.peg_insertion:PegInsertionSimMarkerFLowEnv",
)

register(
    id="LongOpenLockRandPointFlowEnv-v1",
    entry_point="envs.long_open_lock:LongOpenLockRandPointFlowEnv",
)
