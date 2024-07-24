from typing import List, Optional, Union, Any, Callable, Tuple

class Environment:
    def __init__(self):
        pass


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

