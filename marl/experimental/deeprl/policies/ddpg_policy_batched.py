from typing import List

import torch

from marl.experimental.deeprl.policies.ddpg_policy import DDPGPolicy
from marl.experimental.deeprl.policies.networks import my_device
from marl.experimental.deeprl.utils.replay_buffer import Transition


class DDPGPolicyBatched(DDPGPolicy):
    """
    The same as DDPG, but uses vectorized training
    """

    def _update_network(self, batch: List[Transition]):
        assert isinstance(batch, List)
        assert isinstance(batch[0], Transition)
        assert isinstance(batch[0].action, torch.Tensor)

        # [batch_size, state_size]
        states = torch.stack([tr.state for tr in batch]).squeeze(1).squeeze(1)  # remove 1s [batch_size, 1, 1, data_s]
        new_states = torch.stack([tr.new_state for tr in batch]).squeeze(1).squeeze(1)
        # [batch_size, action_size]
        actions = torch.stack([tr.action for tr in batch]).squeeze(1).squeeze(1)
        # [batch_size, 1]
        rewards = torch.tensor([tr.reward for tr in batch], dtype=torch.float, device=my_device()).unsqueeze(-1)

        self.reset()

        # First, compute the actor loss
        actions_actor = self.actor.forward(states)
        action_vals = self.critic_t.forward(torch.cat((states, actions_actor), dim=1))  # TODO critic_t here??

        # actor loss: the higher the action_value, the lower its loss value is => suppress bad actions
        all_actor_losses = -action_vals.mean()

        # Second, compute the critic loss
        with torch.no_grad():
            next_actions = self.actor_t.forward(new_states)
            next_action_vals = self.critic_t.forward(torch.cat((new_states, next_actions), dim=1))

        # Bellman eq. here
        target_vals = self.gamma * next_action_vals + rewards

        # TODO why is this correct?
        # compute what the critic was actually saying
        remembered_action_vals = self.critic.forward(torch.cat((states, actions), dim=1))

        all_critic_losses = self.criterion(remembered_action_vals, target_vals).mean()

        # actor training here
        self.actor.zero_grad()
        all_actor_losses.backward()
        self.actor_optimizer.step()

        # critic training
        self.critic.zero_grad()
        all_critic_losses.backward()
        self.critic_optimizer.step()

        self.num_learns += 1
        self.last_actor_loss = all_actor_losses.item()
        self.last_critic_loss = all_critic_losses.item()
        self.track_targets(self.tau)




