class BasePolicy:
    """Base class for policies.

    Attributes:
        env (pettingzoo.utils.env.AECEnv): Environment used by policy.
    """

    def __init__(self, env):
        """Initialize policy from environment.

        Args:
            env (pettingzoo.utils.env.AECEnv): Environment on which to base
                policy.
        """
        self.env = env

    def action(self, observation, agent):
        """Retrieve action for agent based on observation.

        Args:
            observation: Observation for agent.
            agent (str): Agent for which to retrieve action.
        """
        raise NotImplementedError
