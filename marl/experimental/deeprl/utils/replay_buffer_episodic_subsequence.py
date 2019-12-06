import random
from typing import Optional, List, Tuple

import torch

from marl.experimental.deeprl.policies.networks import my_device
from marl.experimental.deeprl.utils.replay_buffer import Transition
from marl.experimental.deeprl.utils.replay_buffer_episodic import ReplayBufferEpisodic


class ReplayBufferEpisodicSubsequence(ReplayBufferEpisodic):
    """
    This buffer operates on episodes, but it samples sub-sequences of episodes of up to sequence_length.
    """

    buff_sequence_length: Optional[int]

    def __init__(self,
                 buffer_size: int,
                 state_size: int,
                 num_actions: int,
                 buff_sequence_length: Optional[int],
                 _run=None):
        """
        Args:
            buffer_size:
            state_size:
            num_actions:
            buff_sequence_length: if None, the whole episodes will be used
            _run:
        """
        super().__init__(buffer_size=buffer_size,
                         state_size=state_size,
                         num_actions=num_actions,
                         _run=_run)

        assert buff_sequence_length is None or buff_sequence_length > 1, 'buff_seq_length either None or > 1'
        self.buff_sequence_length = buff_sequence_length

    def _get_max_start_step(self, ep_length: int) -> int:
        """
        Returns: max position in the episode where to start in order to get sequence_length of steps,
            if there is not enough steps or sequence_length not specified, return 0.
        """
        if self.buff_sequence_length is None:
            return 0
        return max(ep_length - self.buff_sequence_length, 0)

    @property
    def sequence_length(self) -> Optional[int]:
        """
        Returns: sequence length that will be used for training
            (either chosen as config.buff_seq_length or determined from the buffer)
        """
        if self.buff_sequence_length is not None:
            return self.buff_sequence_length
        if self.num_items == 0:
            print(f'WARNING: accessed the sequence length which is not known yet')
            return None
        return len(self.buffer[0])

    def sample(self, batch_size: int):
        """
        Sample batch_size of sub-episodes from the buffer.

        Length of each sub-episode should be up to the sequence_length
        """
        assert batch_size > 0
        batch = []

        for sample in range(batch_size):
            # pick the episode randomly
            choice = random.randint(0, self.num_items - 1)
            episode = self.buffer[choice]

            # determine interval of possible starts of the sampled sequence: <0, max_start_step>
            ep_length = len(episode)
            max_start_step = self._get_max_start_step(ep_length)
            actual_sub_ep_len = ep_length - max_start_step

            # pick random start of the sub-episode
            start = random.randint(0, max_start_step)
            sub_episode = episode[start: start + actual_sub_ep_len]

            # TODO note that the sub-episodes might not end with done=True,
            #  adding this would change the data in the buffer...
            # sub_episode[-1].done = True

            batch.append(sub_episode)

        return batch

    @staticmethod
    def _find_max_len(batch: List[List[Transition]]) -> int:
        max_len = 0
        for sequence in batch:
            max_len = max(max_len, len(sequence))
        return max_len

    """Here, just a safe reading of potentially variable length episodes"""
    @staticmethod
    def _get_state(episode: List[Transition], pos: int) -> torch.Tensor:
        if pos >= len(episode):
            return torch.zeros_like(episode[0].state)
        return episode[pos].state

    @staticmethod
    def _get_next_state(episode: List[Transition], pos: int) -> torch.Tensor:
        if pos >= len(episode):
            return torch.zeros_like(episode[0].new_state)
        return episode[pos].new_state

    @staticmethod
    def _get_action(episode: List[Transition], pos: int) -> torch.Tensor:
        if pos >= len(episode):
            return torch.zeros_like(episode[0].action)
        return episode[pos].action

    @staticmethod
    def _get_reward(episode: List[Transition], pos: int) -> float:

        if pos >= len(episode):
            return 0
        return episode[pos].reward

    @staticmethod
    def _get_not_done(episode: List[Transition], pos: int) -> float:
        """
        Returns: 0 if done, 1 if not yet
        """

        if pos >= len(episode) or episode[pos].done:
            return 0
        return 1

    @staticmethod
    def _shift_not_done_flag(not_dones: torch.Tensor):
        """
        Shift the not_done flag one step right in the tensor (otherwise the last step is not considered for training)
        """
        nd = (not_dones == 0.0).nonzero()

        if nd.numel() == 0:
            return

        not_dones[nd[0][0], nd[0][1], nd[0][2]] = 1.0

    def compose_batch_tensors(self, batch: List[List[Transition]]) -> \
            Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """Convert list of sequences to batch of sequences, pad everything to the same max_seq_length.

        Args:
            batch: result of the sample method (list of sequences)

        Returns [states, actions, next_states, rewards, not_dones];
            -each thing is a list of Tensors, one Tensor for one position in the sequence (all sequences),
            -the tensor dimensions are: [batch_size, 1, data_size]
        """
        assert isinstance(batch, List)
        assert isinstance(batch[0], List)
        assert isinstance(batch[0][0], Transition)

        assert isinstance(batch[0][0].action, torch.Tensor)  # TODO support also integers here

        assert len(batch[0][0].state.shape) == 3, 'expects the states to be tensors of shape [1, 1, state_size]'
        assert len(batch[0][0].new_state.shape) == 3, 'expects the states to be tensors of shape [1, 1, state_size]'
        assert len(batch[0][0].action.shape) == 3, 'expects the actions to be tensors of shape [1, 1, num_actions]'

        all_states = []
        all_next_states = []
        all_actions = []
        all_rewards = []
        all_not_dones = []

        max_seq_len = ReplayBufferEpisodicSubsequence._find_max_len(batch)

        # go through each step of the episode(s) and stack tensors from the same step to batch_size dim
        for sequence_pos in range(max_seq_len):

            # [batch_size, 1, data_size]
            st = torch.stack([self._get_state(episode, sequence_pos) for episode in batch]).squeeze(1)  # remove old batch_s=1
            next_st = torch.stack([self._get_next_state(episode, sequence_pos) for episode in batch]).squeeze(1)

            # [batch_size, 1]
            act = torch.stack([self._get_action(episode, sequence_pos) for episode in batch]).squeeze(1)
            rew = torch.tensor([self._get_reward(episode, sequence_pos) for episode in batch], device=my_device()).view(-1, 1, 1)

            # invert the flag for convenience
            not_dones = torch.tensor([self._get_not_done(episode, sequence_pos) for episode in batch],
                                     dtype=torch.float32, device=my_device()).view(-1, 1, 1)
            self._shift_not_done_flag(not_dones)  # shift one step further

            all_states.append(st)
            all_next_states.append(next_st)
            all_actions.append(act)
            all_rewards.append(rew)
            all_not_dones.append(not_dones)

        # TODO in case the buff_seq_length is None, the reward from the last time step (of the rollout)
        #  is not handled correctly by the RDDPG algorithms!
        # artificially introduce not DONE flag at the end of each sub-sequence
        all_not_dones[-1] = all_not_dones[-1] * 0

        return all_states, all_actions, all_next_states, all_rewards, all_not_dones

    def compose_batch_sequence_tensors(self, batch: List[List[Transition]]) -> \
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert list of sequences to single batch of sequences, pad everything to the same max_seq_length.

        Args:
            batch: result of the sample method (list of sequences)

        Returns [states, actions, next_states, rewards, not_dones];
            -each thing is a Tensor, which incorporates a whole batch of (possibly padded) rollouts
            -the tensor dimensions are: [batch_size, seq_len, data_size]
        """

        # TODO quite inefficient
        all_states, all_actions, all_next_states, all_rewards, all_not_dones = self.compose_batch_tensors(batch)

        # Now that all sequences have been marshalled and aligned, compose them together
        all_states = torch.cat(all_states, dim=1)
        all_actions = torch.cat(all_actions, dim=1)
        all_next_states = torch.cat(all_next_states, dim=1)
        all_rewards = torch.cat(all_rewards, dim=1)
        all_not_dones = torch.cat(all_not_dones, dim=1)

        return all_states, all_actions, all_next_states, all_rewards, all_not_dones





