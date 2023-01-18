import numpy as np


def policy(**kwargs):
    policy = GreedyPolicy(**kwargs)
    return policy


class GreedyPolicy:
    """Locally optimal policy using a greedy approach.


    This policy computes the reward for every action in every step and chooses
    the action with the highest reward.

    Note that this is only locally optimal and not globally. If the cost
    of cheating is low then the policy will degenerate and always resample
    the same point.
    """

    def __init__(self, env):
        self.env = env

    def action(self, observation, agent):
        """Returns the action with the highest reward."""
        if self.env.terminations[agent] or self.env.truncations[agent]:
            # Agent is dead, the only valid action is None.
            return None
        best_reward = -np.inf
        best_action = None
        for action, point in enumerate(self.env.points):
            reward = self.env.reward(
                self.env.collectors[agent],
                point,
            )
            if reward > best_reward:
                best_reward = reward
                best_action = action
        return best_action
