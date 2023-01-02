from itertools import cycle


class DummyPolicy:
    """Dummy policy that cycles through all actions."""

    def __init__(self, env):
        self.env = env
        self.actions = self._get_possible_actions(env)

    def _get_possible_actions(self, env):
        actions = {}
        for agent in env.possible_agents:
            action_space = env.action_spaces[agent]
            actions[agent] = cycle(range(action_space.n))
        return actions

    def action(self, observation, agent):
        """Returns the next action."""
        if self.env.terminations[agent] or self.env.truncations[agent]:
            # Agent is dead, the only valid action is None.
            return None
        action = next(self.actions[agent])
        return action
