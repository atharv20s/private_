import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np


class ReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size=200000):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.states = np.zeros((max_size, state_dim), dtype=np.float32)
        self.actions = np.zeros((max_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros((max_size, 1), dtype=np.float32)
        self.next_states = np.zeros((max_size, state_dim), dtype=np.float32)
        self.dones = np.zeros((max_size, 1), dtype=np.float32)

    def add(self, state, action, reward, next_state, done):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(self.states[idx]),
            torch.FloatTensor(self.actions[idx]),
            torch.FloatTensor(self.rewards[idx]),
            torch.FloatTensor(self.next_states[idx]),
            torch.FloatTensor(self.dones[idx]),
        )


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.out(x))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.q1_fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.q1_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q1_out = nn.Linear(hidden_dim, 1)

        self.q2_fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.q2_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q2_out = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        xu = torch.cat([state, action], dim=1)

        q1 = F.relu(self.q1_fc1(xu))
        q1 = F.relu(self.q1_fc2(q1))
        q1 = self.q1_out(q1)

        q2 = F.relu(self.q2_fc1(xu))
        q2 = F.relu(self.q2_fc2(q2))
        q2 = self.q2_out(q2)
        return q1, q2

    def q1(self, state, action):
        xu = torch.cat([state, action], dim=1)
        q1 = F.relu(self.q1_fc1(xu))
        q1 = F.relu(self.q1_fc2(q1))
        return self.q1_out(q1)


class TD3Agent:
    def __init__(self, config):
        self.config = config
        self.device = config.DEVICE

        self.actor = Actor(config.STATE_DIM, config.ACTION_DIM, config.HIDDEN_DIM).to(self.device)
        self.actor_target = Actor(config.STATE_DIM, config.ACTION_DIM, config.HIDDEN_DIM).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic(config.STATE_DIM, config.ACTION_DIM, config.HIDDEN_DIM).to(self.device)
        self.critic_target = Critic(config.STATE_DIM, config.ACTION_DIM, config.HIDDEN_DIM).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Lower weight decay helps stabilize actor training
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.ACTOR_LR, weight_decay=1e-5)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.CRITIC_LR, weight_decay=1e-5)

        self.replay_buffer = ReplayBuffer(config.STATE_DIM, config.ACTION_DIM, config.BUFFER_SIZE)
        self.total_steps = 0
        self.total_updates = 0
        self.actor_losses = []
        self.critic_losses = []
        self.gradient_steps = 0  # Track updates for potential LR scheduling

    def select_action(self, state, noise=0.0):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.actor(state_tensor).cpu().numpy()[0]
        if noise > 0:
            action = action + np.random.normal(0, noise, size=action.shape)
        return np.clip(action, -1.0, 1.0)

    def train(self, batch_size):
        if self.replay_buffer.size < batch_size:
            return None, None

        self.total_updates += 1

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        with torch.no_grad():
            noise = (torch.randn_like(actions) * self.config.POLICY_NOISE).clamp(
                -self.config.NOISE_CLIP, self.config.NOISE_CLIP
            )
            next_actions = (self.actor_target(next_states) + noise).clamp(-1.0, 1.0)
            target_q1, target_q2 = self.critic_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards + (1 - dones) * self.config.GAMMA * target_q

        current_q1, current_q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # Stronger gradient clipping for stability - prevents large weight updates
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
        self.critic_optimizer.step()

        actor_loss_val = None
        if self.total_updates % self.config.POLICY_FREQ == 0:
            actor_loss = -self.critic.q1(states, self.actor(states)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            # Stronger gradient clipping for actor to prevent divergence
            nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.3)
            self.actor_optimizer.step()
            actor_loss_val = actor_loss.item()

            for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
                target_param.data.copy_(self.config.TAU * param.data + (1 - self.config.TAU) * target_param.data)
            for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_param.data.copy_(self.config.TAU * param.data + (1 - self.config.TAU) * target_param.data)

        self.actor_losses.append(actor_loss_val if actor_loss_val is not None else np.nan)
        self.critic_losses.append(critic_loss.item())
        return actor_loss_val, critic_loss.item()

    def save(self, filepath):
        torch.save(
            {
                "actor_state_dict": self.actor.state_dict(),
                "critic_state_dict": self.critic.state_dict(),
                "actor_target_state_dict": self.actor_target.state_dict(),
                "critic_target_state_dict": self.critic_target.state_dict(),
                "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
                "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
                "total_steps": self.total_steps,
            },
            filepath,
        )

    def load(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        self.actor_target.load_state_dict(checkpoint["actor_target_state_dict"])
        self.critic_target.load_state_dict(checkpoint["critic_target_state_dict"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])
        self.total_steps = checkpoint.get("total_steps", 0)
