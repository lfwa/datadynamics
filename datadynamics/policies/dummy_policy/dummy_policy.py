from itertools import cycle

from datadynamics.policies.base_policy.base_policy import BasePolicy


def policy(**kwargs):
    """Creates a dummy policy for a given environment.

    Returns:
        BasePolicy: Dummy policy.
    """
    policy = DummyPolicy(**kwargs)
    return policy


class DummyPolicy(BasePolicy):
    """Dummy policy that cycles through all actions.

    Compatible with all environments.

    Attributes:
        env (pettingzoo.utils.env.AECEnv): Environment used by policy.
    """

    def __init__(self, env):
        self.env = env
        self._actions = self._get_possible_actions(env)

    def _get_possible_actions(self, env):
        """Retrieve all possible actions for given environment

        Args:
            env (pettingzoo.utils.env.AECEnv): Environment for which to
                retrieve actions.

        Returns:
            dict: Dictionary of action iterators keyed by agent.
        """
        actions = {}
        for agent in env.possible_agents:
            action_space = env.action_spaces[agent]
            actions[agent] = cycle(
                range(action_space.start, action_space.n + action_space.start)
            )
        return actions

    def action(self, observation, agent):
        if self.env.terminations[agent] or self.env.truncations[agent]:
            # Agent is dead, the only valid action is None.
            return None
        action = next(self._actions[agent])
        return action
