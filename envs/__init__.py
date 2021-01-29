'''
Register new environments
'''
from env.env import Env
from env.vec_env import VecEnv
from env.registration import register, make



register(
    env_id='doudizhu',
    entry_point='env.doudizhu:DoudizhuEnv',
)


