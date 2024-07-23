from typing import List, Optional, Union, Any, Callable, Tuple

class Environment:
    def __init__(self, state_space: Optional[List[Any]] = None, action_space: Optional[List[Any]] = None):
        """
        Initialize the environment with optional state and action spaces.

        Args:
            state_space (Optional[List[Any]]): The list of possible states.
            action_space (Optional[List[Any]]): The list of possible actions.
        """
        self.state_space = state_space
        self.action_space = action_space

    def step(self, state: Any, action: Any) -> Tuple[Any, float]:
        """
        Perform an action in the environment from a given state.

        Args:
            state (Any): The current state.
            action (Any): The action to be taken.

        Raises:
            NotImplementedError: This method should be overridden by subclasses.

        Returns:
            Tuple[Any, float]: The next state and the reward received.
        """
        raise NotImplementedError("Subclass Environment needs to implement step()")

    def reset(self) -> Any:
        """
        Reset the environment to the initial state.

        Raises:
            NotImplementedError: This method should be overridden by subclasses.

        Returns:
            Any: The initial state.
        """
        raise NotImplementedError("Subclass Environment needs to implement reset() for resetting episode")

    def is_terminal(self, state: Any) -> bool:
        """
        Check if the given state is a terminal state.

        Args:
            state (Any): The state to check.

        Raises:
            NotImplementedError: This method should be overridden by subclasses.

        Returns:
            bool: True if the state is terminal, False otherwise.
        """
        raise NotImplementedError("Subclass Environment needs to implement is_terminal()")

    def render(self) -> None:
        """
        Render the current state of the environment.

        Raises:
            NotImplementedError: This method should be overridden by subclasses.
        """
        raise NotImplementedError("Subclass Environment needs to implement render() for visualizing the environment")

    def seed(self, seed: Optional[int] = None) -> None:
        """
        Set the random seed for the environment.

        Args:
            seed (Optional[int]): The seed value.
        """
        raise NotImplementedError("Subclass Environment needs to implement seed() for setting the random seed")

    def validate_state_space(self) -> None:
        """
        Validate the state space.

        Raises:
            ValueError: If the state space is invalid.
        """
        if not isinstance(self.state_space, list):
            raise ValueError("State space must be a list")

    def validate_action_space(self) -> None:
        """
        Validate the action space.

        Raises:
            ValueError: If the action space is invalid.
        """
        if not isinstance(self.action_space, list):
            raise ValueError("Action space must be a list")