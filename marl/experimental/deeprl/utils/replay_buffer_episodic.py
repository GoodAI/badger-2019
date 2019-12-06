from typing import List, Union
import torch
from marl.experimental.deeprl.utils.replay_buffer import ReplayBuffer, Transition


class ReplayBufferEpisodic(ReplayBuffer):
    """
    This buffer does not save Transitions, but whole episodes (as lists of transitions).
    Sampling produces batch_size of episodes.
    """
    buffer: List[List[Transition]]
    current_episode: List[Transition]

    def __init__(self,
                 buffer_size: int,
                 state_size: int,
                 num_actions: int,
                 _run=None):
        super().__init__(buffer_size=buffer_size,
                         state_size=state_size,
                         num_actions=num_actions,
                         _run=_run)

        self.current_episode = []

    def remember(self,
                 state: torch.Tensor,
                 action: Union[int, torch.Tensor],
                 new_state: torch.Tensor,
                 reward: float,
                 done: bool):
        """
        Remember one transition, create episode and push to the buffer if done.
        """
        state, action, new_state, reward, done = self._sanitize_inputs(state, action, new_state, reward, done)

        self._check_inputs(state=state, action=action, new_state=new_state, reward=reward, done=done)
        transition = Transition(state, action, new_state, reward, done)

        self.current_episode.append(transition)

        if done:
            if len(self.current_episode) < 2:
                print(f'WARNING: current episode has very small length of {len(self.current_episode)}')

            self.push_to_buffer(self.current_episode[:])  # should deepcopy
            self.current_episode.clear()  # start new episode

