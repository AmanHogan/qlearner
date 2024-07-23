class Environment:
    def __init__(self):
        pass

    def step(self, state, action):
        raise NotImplementedError("Subclass Environment needs to implement step()")

    def reset(self):
        raise NotImplementedError("Subclass Environment needs to implement reset() for resetting episode")

    def is_terminal(self, state):
        raise NotImplementedError("Subclass Environment needs to implement is_terminal()")