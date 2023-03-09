from datadynamics.policies.base_policy.base_policy import BasePolicy


def policy(**kwargs):
    """Creates a RandomPolicy for a given environment.

    Returns:
        BasePolicy: Random policy.
    """
    policy = RandomPolicy(**kwargs)
    return policy


class RandomPolicy(BasePolicy):
    """Policy that returns a random action.

    Compatible with all environments.

    Attributes:
        env (pettingzoo.utils.env.AECEnv): Environment used by policy.
    """

    def __init__(self, env):
        self.env = env

    def action(self, observation, agent):
        if self.env.terminations[agent] or self.env.truncations[agent]:
            # Agent is dead, the only valid action is None.
            return None
        action = self.env.action_space(agent).sample()
        return action
