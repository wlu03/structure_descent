"""
Delphos baseline: DQN agent that builds a utility specification by sequential
Add/Change/Terminate actions over (variable, transformation) tuples.

Reference
---------
Nova, G., Hess, S., van Cranenburgh, S. (2025). "Delphos: A reinforcement
learning framework for assisting discrete choice model specification."
arXiv:2506.06410. https://arxiv.org/abs/2506.06410

The architecture (DQN + experience replay + target network + min-max
normalized reward + Add/Change/Terminate action space over
(variable, transformation) tuples + distributed terminal reward +
window-based early stopping) structurally follows §3 of Nova et al. 2025.

What is replaced (the estimator)
--------------------------------
Delphos's ``delphos_interaction`` in the paper delegates to R/Apollo for the
inner MNL estimation. We replace this call with ``_estimate_from_state``
which:
    1. decodes the Delphos state list [(var, trans), ...] into a
       ``DSLStructure`` using ``ALL_TERMS``,
    2. fits hierarchical weights via ``src.inner_loop.fit_weights``,
    3. returns a single-row pandas DataFrame with the column set Delphos's
       reward function and normalization routines read (LL0, LLC, LLout,
       rho2_0, rho2_C, adjRho2_0, adjRho2_C, AIC, BIC, numParams).

Metrics computed honestly
-------------------------
- ``LL0`` is the equal-probability null, summed per-event from the
  actual choice-set size of each training event (not a constant n_alts).
- ``LLC`` is the constants-only MNL baseline: log-likelihood of a model
  with zero weights (uniform-over-choice-set) on the training batch.
  For this DSL, which carries no alternative-specific constants, LLC
  equals LL0.
- ``LLout`` is the log-likelihood evaluated on the validation batch.
- ``rho2_C`` / ``adjRho2_C`` are derived from ``LLC`` rather than aliased
  to ``rho2_0``.
- ``numParams`` counts the effective hierarchical parameter count
  (global + per-category + per-customer-with-delta terms), not just the
  DSL term count.

Deviations from faithful Delphos (documented)
---------------------------------------------
- ASC slot (var=0) is dropped. Our hierarchical fit does not carry
  alternative-specific constants as a separate slot.
- ``box-cox`` -> fixed ``power(exponent=0.5)``. No learned lambda.
- Specific taste (``spec``) dropped. Our events share coefficients across
  alternatives already.
- Covariates (``cov``) dropped. No categorical covariate interaction
  support in the initial port.
- No behavioral sign check (Eq. 17). Delphos itself does not implement
  this in its own ``reward_function`` — confirmed from paper §3.
- ``rewards.csv`` persistence stripped. Replaced with an in-memory dict
  keyed on ``StateManager.encode_state_to_string``.
- Default ``reward_weights = {'AIC': 1}`` matches the paper default.

Interface: implements Baseline / FittedBaseline from src/baselines/base.py.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.special import log_softmax

from ..dsl import ALL_TERMS, DSLStructure, DSLTerm, build_structure_features
from ..inner_loop import HierarchicalWeights, fit_weights
from .base import BaselineEventBatch, FittedBaseline


# =============================================================================
# DQNetwork  (verbatim from Delphos agent.py L34-55)
# =============================================================================

class DQNetwork(nn.Module):
    """
    Deep Q-Network with a variable number of hidden layers.

    Args:
        input_size (int): Size of the input tensor.
        output_size (int): Size of the output tensor.
        hidden_layers (List[int], optional): List of hidden layer sizes.
            Defaults to [128, 64].
    """
    def __init__(self, input_size: int, output_size: int,
                 hidden_layers: List[int] = [128, 64]) -> None:
        super().__init__()
        layers = []
        last_size = input_size
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(last_size, hidden_size))
            layers.append(nn.ReLU())
            last_size = hidden_size
        layers.append(nn.Linear(last_size, output_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# =============================================================================
# ReplayBuffer  (verbatim from Delphos agent.py L58-92)
# =============================================================================

class ReplayBuffer:
    """Replay buffer for storing and sampling past experiences."""
    def __init__(self, max_size: int) -> None:
        self.buffer = deque(maxlen=max_size)

    def add(self, transition: Tuple) -> None:
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> Any:
        if len(self.buffer) < batch_size:
            return None
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)


# =============================================================================
# StateManager  (verbatim from Delphos agent.py L95-328)
# =============================================================================

class StateManager:
    """
    Manages the encoding and decoding of state representations for model
    specifications.
    """
    def __init__(self, state_space_params: Dict[str, Any]) -> None:
        default_trans = ['linear', 'log', 'box-cox']
        default_taste = ['generic', 'specific']
        self.state_space_params = {
            'num_vars': state_space_params.get('num_vars', 1),
            'transformations': state_space_params.get('transformations', default_trans),
            'taste': state_space_params.get('taste', default_taste),
            'covariates': state_space_params.get('covariates', [])
        }
        self.transformation_codes = {0: 'none', 1: 'linear', 2: 'log', 3: 'box-cox'}
        self.inverse_mapping = {v: k for k, v in self.transformation_codes.items()}

    def get_state_length(self) -> int:
        asc = 1
        att = self.state_space_params['num_vars'] - asc
        att_ = 4
        taste_ = 2
        cov_ = len(self.state_space_params['covariates']) + 1
        if 'specific' in self.state_space_params['taste']:
            return (asc * cov_) + (att * att_ * taste_ * cov_)
        else:
            return (asc * cov_) + (att * att_)

    def encode_state_to_vector(self, state: List[Tuple]) -> torch.FloatTensor:
        state_vector = np.zeros(self.get_state_length())
        transformations = self.state_space_params['transformations']
        covariates = self.state_space_params['covariates']
        num_transformations = 4
        num_specific = 2
        num_covariates = len(covariates) + 1
        if not 'specific' in self.state_space_params['taste']:
            asc_offset = 1
            for var, trans in state:
                if var == 0 and trans == 'linear':
                    state_vector[0] = 1
                else:
                    trans_idx = self.inverse_mapping.get(trans, 0)
                    attr_idx = (var - 1) * num_transformations
                    state_vector[asc_offset + attr_idx + trans_idx] = 1
        elif 'specific' in self.state_space_params['taste'] and not covariates:
            for var, trans, spec in state:
                if var == 0 and trans == 'linear' and spec == 0:
                    state_vector[0] = 1
                else:
                    attr_idx = (var - 1) * (num_transformations * num_specific)
                    trans_idx = self.inverse_mapping.get(trans, 0)
                    spec_idx = spec
                    total = attr_idx + (1 - spec_idx) * trans_idx + spec_idx * (num_transformations + trans_idx)
                    state_vector[total] = 1
        else:
            for var, trans, spec, cov in state:
                if var == 0 and trans == 'linear' and spec == 0:
                    state_vector[cov] = 1
                else:
                    asc_offset = 2
                    attr_idx = (var - 1) * (num_transformations * num_specific * num_covariates)
                    trans_idx = self.inverse_mapping.get(trans, 0)
                    spec_idx = spec
                    cov_idx = cov
                    total = asc_offset + attr_idx + (1 - spec_idx) * trans_idx + spec_idx * (num_transformations + trans_idx) + cov_idx * (num_transformations * num_specific)
                    state_vector[total] = 1
        return torch.FloatTensor(state_vector)

    def encode_state_to_string(self, state: List[Tuple]) -> str:
        representation = ['000'] * self.state_space_params['num_vars']
        covariates = self.state_space_params['covariates']
        use_specific = 'specific' in self.state_space_params['taste']
        for entry in state:
            if use_specific and covariates:
                var, trans, spec, cov = entry
            elif use_specific:
                var, trans, spec = entry
                cov = 0
            else:
                var, trans = entry
                spec, cov = 0, 0
            trans_code = self.inverse_mapping.get(trans, 0)
            representation[var] = f"{trans_code}{spec}{cov}".zfill(3)
        return '_'.join(representation)

    def decode_string_to_state(self, representation: str) -> List[Tuple]:
        covariates = self.state_space_params['covariates']
        use_specific = 'specific' in self.state_space_params['taste']
        rep_list = representation.split('_')
        state = []
        for idx, code in enumerate(rep_list):
            if code == '000':
                continue
            trans_code = int(code[0])
            spec = int(code[1]) if use_specific else 0
            cov = int(code[2]) if covariates else 0
            trans = self.transformation_codes.get(trans_code, 'none')
            if use_specific and covariates:
                state.append((idx, trans, spec, cov))
            elif use_specific:
                state.append((idx, trans, spec))
            else:
                state.append((idx, trans))
        return state

    def mask_invalid_actions(self, state: List[Tuple],
                              action_space: List[Tuple]) -> List[Tuple]:
        transformations = self.state_space_params['transformations']
        covariates = self.state_space_params['covariates']
        taste = self.state_space_params['taste']
        use_specific = 'specific' in taste
        asc_added = any(var == 0 for var, *rest in state)
        valid_actions = []
        add_current_attributes = set()
        change_current_attributes = set()
        if not use_specific and not covariates:
            add_current_attributes = {var for var, trans in state}
            change_current_attributes = {(var, trans) for var, trans in state}
        elif use_specific and not covariates:
            add_current_attributes = {var for var, trans, spec in state}
            change_current_attributes = {(var, trans, spec) for var, trans, spec in state}
        elif use_specific and covariates:
            add_current_attributes = {var for var, trans, spec, cov in state}
            change_current_attributes = {(var, trans, spec, cov) for var, trans, spec, cov in state}
        seen_asc_action = False
        for action in action_space:
            act_type = action[0]
            if act_type == 'terminate':
                valid_actions.append(action)
                continue
            var = action[1]
            trans = action[2]
            if var == 0:
                if trans != 'linear':
                    continue
                if use_specific and len(action) > 3 and action[3] != 0:
                    continue
                if asc_added or seen_asc_action:
                    continue
                valid_actions.append(action)
                seen_asc_action = True
                continue
            if not use_specific and not covariates:
                if act_type == 'add' and var not in add_current_attributes:
                    valid_actions.append(action)
                elif act_type == 'change' and (var, trans) not in change_current_attributes:
                    valid_actions.append(action)
            elif use_specific and not covariates:
                spec = action[3]
                if act_type == 'add' and var not in add_current_attributes:
                    valid_actions.append(action)
                elif act_type == 'change' and (var, trans, spec) not in change_current_attributes:
                    valid_actions.append(action)
            elif use_specific and covariates:
                spec = action[3]
                cov = action[4]
                if act_type == 'add' and var not in add_current_attributes:
                    valid_actions.append(action)
                elif act_type == 'change' and (var, trans, spec, cov) not in change_current_attributes:
                    valid_actions.append(action)
        return valid_actions


# =============================================================================
# DQNLearner  (preserves the full Delphos algorithm; swaps the estimator)
# =============================================================================

# Columns Delphos's reward_function / normalize_reward_metric read
_REWARD_COLUMNS = [
    "numParams", "successfulEstimation", "LL0", "LLC", "LLout",
    "rho2_0", "adjRho2_0", "rho2_C", "adjRho2_C", "AIC", "BIC",
]


class DQNLearner:
    """
    DQN agent for model specification with experience replay, candidate
    tracking, and early stopping. Preserves Delphos's algorithm verbatim;
    the environment call ``delphos_interaction`` is replaced with a user-
    supplied ``estimate_fn(state) -> pd.DataFrame``.

    Only changes relative to Delphos's ``DQNLearner``:
      * No disk paths / rewards.csv persistence.
      * ``estimate_fn`` parameter replaces the R/Apollo call.
      * Logging is silent (no file handlers, no plots).
    """
    def __init__(
        self,
        state_space_params: Dict[str, Any],
        num_episodes: int,
        estimate_fn,
        hidden_layers: Tuple[int, ...] = (128, 64),
        discount_factor: float = 0.9,
        learning_rate: float = 0.001,
        buffer_size: int = 10000,
        batch_size: int = 64,
        target_update_freq: int = 10,
        patience: int = 100,
        early_stop_window: int = 500,
        early_stop_tolerance: float = 0.001,
        min_percentage: float = 0.5,
        reward_weights: Optional[Dict[str, float]] = None,
        reward_distribution: str = 'exponential',
        epsilon_min: float = 0.01,
        seed: int = 0,
    ) -> None:
        # ---- preserve Delphos init semantics, minus disk IO ----
        self.state_space_params = state_space_params
        self.num_episodes = num_episodes
        self.discount_factor = discount_factor
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.epsilon = 1.0
        self.epsilon_min = float(epsilon_min)
        self.epsilon_decay = 1.0 / max(num_episodes, 1)
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.state_manager = StateManager(state_space_params)

        # Reward metrics and weights (Delphos L402-407)
        self.reward_weights = reward_weights or {'AIC': 1.0}
        default_metric_directions = {
            'AIC': 'minimize', 'BIC': 'minimize',
            'adjRho2_0': 'maximize', 'rho2_0': 'maximize',
            'rho2_C': 'maximize', 'adjRho2_C': 'maximize',
            'LLout': 'maximize'
        }
        self.metric_directions = {
            metric: default_metric_directions.get(metric, 'maximize')
            for metric in self.reward_weights
        }
        self.metric = list(self.reward_weights.keys())[0]
        self.reward_distribution = reward_distribution

        # In-memory replacement for rewards.csv (Delphos L410-415)
        self._estimate_cache: Dict[str, pd.DataFrame] = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self._estimate_fn = estimate_fn

        # Action space
        self.action_space, self.action_to_index = self.define_action_space()

        self.action_log: List[dict] = []
        self.buffer_log: List[dict] = []
        self.training_log: List[dict] = []

        # Policy/target nets
        input_size = self.state_manager.get_state_length()
        output_size = len(self.action_space)
        torch.manual_seed(seed)
        self.policy_net = DQNetwork(input_size, output_size, list(hidden_layers))
        self.target_net = DQNetwork(input_size, output_size, list(hidden_layers))
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Candidate tracking (Delphos L452-461)
        self.best_candidates = {
            metric: {
                'value': -np.inf if self.metric_directions[metric] == 'maximize' else np.inf,
                'episode': -1,
                'representation': None,
            }
            for metric in self.reward_weights
        }
        self.LL0: Optional[float] = None
        self.current_best_history: List[dict] = []
        self.patience = patience
        self.early_stop_window = early_stop_window
        self.early_stop_tolerance = early_stop_tolerance
        self.min_percentage = min_percentage
        self.min_episodes_before_stop = self.num_episodes * self.min_percentage
        self.no_improvement_count = 0
        self.model_set: set = set()

        # seed python/numpy rngs for determinism of epsilon-greedy choices
        random.seed(seed)
        np.random.seed(seed)

    # ---- verbatim: define_action_space (Delphos L471-521) ----
    def define_action_space(self) -> Tuple[List[Tuple], Dict[Tuple, int]]:
        actions = [('terminate',)]
        for var in range(self.state_space_params['num_vars']):
            if var == 0:
                if 'specific' in self.state_space_params['taste'] and self.state_space_params['covariates'] != []:
                    for cov in range(len(self.state_space_params['covariates']) + 1):
                        actions.append(('add', var, 'linear', 0, cov))
                        actions.append(('change', var, 'linear', 0, cov))
                elif 'specific' in self.state_space_params['taste'] and self.state_space_params['covariates'] == []:
                    for specific_flag in [0, 1]:
                        actions.append(('add', var, 'linear', specific_flag))
                        actions.append(('change', var, 'linear', specific_flag))
                else:
                    # non-specific, no covariates -> ASC as a plain (var=0, 'linear') tuple
                    actions.append(('add', var, 'linear'))
                    actions.append(('change', var, 'linear'))
            else:
                for transformation in self.state_space_params['transformations']:
                    if self.state_space_params['covariates']:
                        if 'specific' in self.state_space_params['taste']:
                            for specific_flag in [0, 1]:
                                for cov in range(len(self.state_space_params['covariates']) + 1):
                                    actions.append(('add', var, transformation, specific_flag, cov))
                                    actions.append(('change', var, transformation, specific_flag, cov))
                        else:
                            for cov in range(len(self.state_space_params['covariates']) + 1):
                                actions.append(('add', var, transformation, 0, cov))
                                actions.append(('change', var, transformation, 0, cov))
                    else:
                        if 'specific' in self.state_space_params['taste']:
                            for specific_flag in [0, 1]:
                                actions.append(('add', var, transformation, specific_flag))
                                actions.append(('change', var, transformation, specific_flag))
                        else:
                            actions.append(('add', var, transformation))
                            actions.append(('change', var, transformation))
        action_to_index = {action: idx for idx, action in enumerate(actions)}
        return actions, action_to_index

    # ---- verbatim: select_action_index (Delphos L523-546) ----
    def select_action_index(self, state: List[Tuple]) -> int:
        valid_actions = self.state_manager.mask_invalid_actions(state, self.action_space)
        valid_action_indices = [self.action_to_index[action] for action in valid_actions]
        if np.random.rand() < self.epsilon:
            return random.choice(valid_action_indices)
        else:
            with torch.no_grad():
                state_vector = self.state_manager.encode_state_to_vector(state)
                q_values = self.policy_net(state_vector)
                q_values_masked = torch.full((len(self.action_space),), float('-inf'))
                q_values_masked[valid_action_indices] = q_values[valid_action_indices]
                index = torch.argmax(q_values_masked).item()
                return index

    # ---- verbatim: apply_action (Delphos L548-591) ----
    def apply_action(self, state: List[Tuple], action_index: int) -> Tuple[List[Tuple], bool]:
        covariates = self.state_space_params['covariates']
        taste = self.state_space_params['taste']
        use_specific = 'specific' in taste

        state = list(state)
        action = self.action_space[action_index]

        if action[0] == 'terminate':
            return state, True

        if use_specific and covariates == []:
            action_type, var, trans, spec = action
            if action_type == 'add':
                state.append((var, trans, spec))
            elif action_type == 'change':
                state = [(v, t, s) if v != var else (v, trans, spec) for v, t, s in state]

        if use_specific and covariates != []:
            action_type, var, trans, spec, cov = action
            if action_type == 'add':
                state.append((var, trans, spec, cov))
            elif action_type == 'change':
                state = [(v, t, s, c) if v != var else (v, trans, spec, cov) for v, t, s, c in state]

        if not use_specific and covariates == []:
            action_type, var, trans = action
            if action_type == 'add':
                state.append((var, trans))
            elif action_type == 'change':
                state = [(v, t) if v != var else (v, trans) for v, t in state]

        return state, False

    # ---- verbatim: generate_episode (Delphos L593-614) ----
    def generate_episode(self, episode_count: int):
        state, done, episode_steps = [], False, []
        while not done:
            action_index = self.select_action_index(state)
            next_state, done = self.apply_action(state, action_index)
            episode_steps.append((state, action_index, next_state))
            self.action_log.append({
                'episode': episode_count, 'state': state,
                'action': self.action_space[action_index], 'next_state': next_state,
            })
            state = next_state
        return episode_steps, state, done

    # ---- verbatim: update_candidate_tracker (Delphos L616-652) ----
    def update_candidate_tracker(self, episode_count: int,
                                  modelling_outcomes: Dict[str, float],
                                  state: List[Tuple]) -> None:
        best_repr = self.state_manager.encode_state_to_string(state)
        for metric in self.reward_weights:
            if metric in modelling_outcomes:
                value = modelling_outcomes[metric]
                if not isinstance(value, (int, float)) or pd.isna(value) or not np.isfinite(value):
                    continue
                current_best = self.best_candidates[metric]
                direction = self.metric_directions.get(metric, 'maximize')
                improved = (value > current_best['value']
                            if direction == 'maximize'
                            else value < current_best['value'])
                if improved:
                    self.best_candidates[metric] = {
                        'value': value, 'episode': episode_count,
                        'representation': best_repr,
                    }
        snapshot = {'episode': episode_count}
        for metric, info in self.best_candidates.items():
            snapshot[f"{metric}_value"] = info['value']
            snapshot[f"{metric}_repr"] = info['representation']
        self.current_best_history.append(snapshot)

    # ---- verbatim: check_early_stopping (Delphos L654-678) ----
    def check_early_stopping(self, episode_rewards: List[float]) -> bool:
        if len(episode_rewards) < 2 * self.early_stop_window:
            return False
        current_window = episode_rewards[-self.early_stop_window:]
        previous_window = episode_rewards[-2 * self.early_stop_window:-self.early_stop_window]
        current_mean = np.mean(current_window)
        previous_mean = np.mean(previous_window)
        relative_improvement = (current_mean - previous_mean) / (abs(previous_mean) + 1e-8)
        if relative_improvement < self.early_stop_tolerance:
            self.no_improvement_count += 1
        else:
            self.no_improvement_count = 0
        if self.no_improvement_count >= self.patience:
            return True
        return False

    # ---- verbatim: normalize_reward_metric (Delphos L680-724) ----
    def normalize_reward_metric(self, metric: str, value: float) -> float:
        values = [log[metric] for log in self.training_log
                  if metric in log and pd.notna(log[metric])]
        max_val = max(values) if values else 0.0
        min_val = min(values) if values else 0.0
        if not values or self.LL0 is None:
            return 0.0
        norm = 0.0
        if metric == 'LLout':
            if value <= self.LL0 or max_val - self.LL0 == 0:
                return 0.0
            norm = (value - self.LL0) / (max_val - self.LL0)
        elif metric in ['AIC', 'BIC']:
            max_value = -2 * self.LL0
            if value >= max_value or max_val == min_val:
                return 0.0
            norm = (max_val - value) / (max_val - min_val)
        elif metric in ['rho2_0', 'adjRho2_0', 'rho2_C', 'adjRho2_C']:
            if value <= 0:
                return 0.0
            else:
                norm = value
        if np.isfinite(norm) and not pd.isna(norm):
            return float(norm)
        else:
            return 0.0

    # ---- verbatim: reward_function (Delphos L726-758) ----
    def reward_function(self, modelling_outcome: pd.DataFrame) -> float:
        try:
            if modelling_outcome.empty or modelling_outcome['successfulEstimation'].values[0] is False:
                return 0.0
            reward = 0.0
            for metric, weight in self.reward_weights.items():
                if metric in modelling_outcome.columns:
                    raw_value = modelling_outcome[metric].values[0]
                    if not pd.isna(raw_value):
                        normalized_value = self.normalize_reward_metric(metric, raw_value)
                    else:
                        normalized_value = 0.0
                    reward += weight * normalized_value
            return reward
        except Exception:
            return 0.0

    # ---- replaces delphos_interaction: calls user-provided estimator ----
    def delphos_interaction(self, state: List[Tuple]):
        """Replacement for Delphos's delphos_interaction.

        Uses the in-memory cache keyed on encode_state_to_string and calls
        the user-supplied estimate_fn(state) -> pd.DataFrame.
        """
        key = self.state_manager.encode_state_to_string(state)
        if key in self._estimate_cache:
            self.cache_hits += 1
            modelling_outcomes = self._estimate_cache[key]
        else:
            self.cache_misses += 1
            modelling_outcomes = self._estimate_fn(state)
            self._estimate_cache[key] = modelling_outcomes

        if self.LL0 is None and 'LL0' in modelling_outcomes.columns:
            ll0_val = modelling_outcomes['LL0'].values[0]
            if pd.notna(ll0_val) and np.isfinite(ll0_val):
                self.LL0 = float(ll0_val)

        final_reward = self.reward_function(modelling_outcomes)
        # Restrict returned columns to what Delphos's training log expects
        keep_cols = list(set(list(self.reward_weights.keys())
                             + [c for c in _REWARD_COLUMNS
                                if c in modelling_outcomes.columns]))
        reduced = modelling_outcomes[modelling_outcomes.columns.intersection(keep_cols)]
        return final_reward, reduced

    # ---- verbatim: perform_experience_replay (Delphos L814-844) ----
    def perform_experience_replay(self) -> None:
        batch = self.replay_buffer.sample(self.batch_size)
        if not batch:
            return
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)
        state_batch = torch.stack(state_batch)
        action_batch = torch.LongTensor(action_batch).unsqueeze(1)
        reward_batch = torch.FloatTensor(reward_batch)
        next_state_batch = torch.stack(next_state_batch)
        done_batch = torch.BoolTensor(done_batch)

        q_values = self.policy_net(state_batch).gather(1, action_batch)
        next_q_values = self.target_net(next_state_batch).max(1)[0].detach()
        target_q_values = reward_batch + (1 - done_batch.float()) * self.discount_factor * next_q_values

        loss = nn.functional.mse_loss(q_values, target_q_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    # ---- verbatim: train main loop (Delphos L846-928), plots/saves stripped ----
    def train(self) -> int:
        """Train the DQN. Returns number of episodes actually run."""
        episode_count = 0
        episode_rewards: List[float] = []
        self.current_best_history = []
        early_stop_triggered = False

        for episode_count in range(self.num_episodes):
            episode_steps, state, done = self.generate_episode(episode_count)
            # generate_episode only exits with done=True, so interaction always runs.
            final_reward, modelling_outcomes = self.delphos_interaction(state)
            episode_rewards.append(final_reward)
            final_repr = self.state_manager.encode_state_to_string(state)
            self.model_set.add(final_repr)

            L = len(episode_steps)

            if self.reward_distribution == 'uniform':
                inter_reward = final_reward / L if L > 0 else final_reward
                for step_state, step_action, step_next_state in episode_steps:
                    self.replay_buffer.add((
                        self.state_manager.encode_state_to_vector(step_state),
                        step_action, inter_reward,
                        self.state_manager.encode_state_to_vector(step_next_state),
                        done,
                    ))
            elif self.reward_distribution == 'linear':
                for l, (step_state, step_action, step_next_state) in enumerate(episode_steps):
                    inter_reward = (final_reward * (l + 1)) / L if L > 0 else final_reward
                    self.replay_buffer.add((
                        self.state_manager.encode_state_to_vector(step_state),
                        step_action, inter_reward,
                        self.state_manager.encode_state_to_vector(step_next_state),
                        done,
                    ))
            else:
                gamma = self.discount_factor
                for l, (step_state, step_action, step_next_state) in enumerate(episode_steps):
                    discounted_reward = (gamma ** (L - l - 1)) * final_reward if L > 0 else final_reward
                    self.replay_buffer.add((
                        self.state_manager.encode_state_to_vector(step_state),
                        step_action, discounted_reward,
                        self.state_manager.encode_state_to_vector(step_next_state),
                        done,
                    ))

            log = {
                'episode': episode_count, 'specification': final_repr,
                'reward': final_reward, 'epsilon': self.epsilon,
            }
            # modelling_outcomes is a pd.DataFrame here
            if isinstance(modelling_outcomes, pd.DataFrame) and not modelling_outcomes.empty:
                for key in modelling_outcomes.columns:
                    value = modelling_outcomes[key].iloc[0]
                    if key in self.reward_weights:
                        log[key] = value
            self.training_log.append(log)

            metrics_dict: Dict[str, float] = {}
            if isinstance(modelling_outcomes, pd.DataFrame) and not modelling_outcomes.empty:
                for key in modelling_outcomes.columns:
                    metrics_dict[key] = modelling_outcomes[key].iloc[0]
            self.update_candidate_tracker(episode_count, metrics_dict, state)
            self.perform_experience_replay()

            if episode_count % self.target_update_freq == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

            self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_decay)

            if episode_count >= self.min_episodes_before_stop:
                if self.check_early_stopping(episode_rewards):
                    early_stop_triggered = True
                    break

        total_episodes = episode_count + 1
        return total_episodes


# =============================================================================
# State <-> DSLStructure bridge  (the new estimator)
# =============================================================================


def _state_to_structure(state: List[Tuple]) -> DSLStructure:
    """Decode a Delphos state list into a DSLStructure.

    Mapping: var=0 is Delphos's ASC slot and is skipped. For var >= 1,
    base_name = ALL_TERMS[var - 1]. Transformation -> DSL term form:
      * 'linear'  -> DSLTerm(base_name)
      * 'log'     -> DSLTerm('log_transform', args=[base_name])
      * 'box-cox' -> DSLTerm('power', args=[base_name], kwargs={'exponent': 0.5})
    Unknown transformations are ignored. If the decoded structure is empty,
    fall back to DSLStructure(['routine']) so fit_weights is well-defined.
    """
    terms: List[DSLTerm] = []
    for entry in state:
        if len(entry) >= 2:
            var, trans = entry[0], entry[1]
        else:
            continue
        if var == 0:
            continue  # ASC — dropped (documented deviation)
        if not (1 <= var <= len(ALL_TERMS)):
            continue
        base_name = ALL_TERMS[var - 1]
        if trans == 'linear':
            terms.append(DSLTerm(name=base_name))
        elif trans == 'log':
            terms.append(DSLTerm(name='log_transform', args=[base_name]))
        elif trans == 'box-cox':
            terms.append(DSLTerm(name='power', args=[base_name], kwargs={'exponent': 0.5}))
        # 'none' or other -> skip
    if not terms:
        return DSLStructure(['routine'])
    return DSLStructure(terms)


def _features_for_structure(
    structure: DSLStructure,
    batch: BaselineEventBatch,
) -> List[np.ndarray]:
    names = batch.base_feature_names
    return [
        build_structure_features(structure, base_feats, names)
        for base_feats in batch.base_features_list
    ]


def _loglik_from_weights(
    features_list: List[np.ndarray],
    chosen_indices: List[int],
    customer_ids: List[str],
    categories: List[str],
    weights: HierarchicalWeights,
) -> float:
    """Signed log-likelihood (LL, NOT negative NLL)."""
    total = 0.0
    for feats, chosen, cid, cat in zip(
        features_list, chosen_indices, customer_ids, categories
    ):
        w = weights.get_weights(cid, cat)
        lp = log_softmax(feats @ w)
        total += float(lp[chosen])
    return total


def _null_loglik(batch: BaselineEventBatch) -> float:
    """Equal-probability null log-likelihood, summed per-event.

    LL0 = sum_i log(1 / |A(s_i)|), respecting variable choice-set sizes.
    """
    total = 0.0
    for feats in batch.base_features_list:
        n_alts_i = int(feats.shape[0])
        if n_alts_i <= 0:
            return float('nan')
        total += float(np.log(1.0 / n_alts_i))
    return total


def _effective_n_params(structure: DSLStructure, weights: HierarchicalWeights) -> int:
    """Hierarchical param count: global + per-category + per-customer-with-delta.

    Matches the flat layout used by inner_loop.fit_weights:
    n_terms * (1 + n_cats + n_custs_with_delta).
    """
    n_terms = len(structure.terms)
    n_cats = len(weights.theta_c)
    n_delta_i = len(weights.delta_i)
    return int(n_terms * (1 + n_cats + n_delta_i))


def _estimate_from_state(
    state: List[Tuple],
    train: BaselineEventBatch,
    val: Optional[BaselineEventBatch] = None,
    fit_kwargs: Optional[dict] = None,
) -> pd.DataFrame:
    """Replacement for Delphos's delphos_interaction.

    Decodes the Delphos state into a DSLStructure, fits hierarchical
    weights on the training batch, and returns a single-row DataFrame
    with the column set Delphos's reward_function reads.

    Honest metric accounting
    ------------------------
    - ``LL0`` is computed per-event from each event's actual choice-set
      size (variable |A(s_i)|), not a constant n_alts.
    - ``LLC`` is the constants-only MNL log-likelihood on the training
      batch. Our DSL carries no alternative-specific constants, so LLC
      reduces to LL0.
    - ``LLout`` is the log-likelihood on the validation batch if one is
      provided; otherwise NaN.
    - ``rho2_C`` and ``adjRho2_C`` are derived from LLC (not aliased to
      rho2_0).
    - ``numParams`` uses the effective hierarchical count, not
      ``len(structure.terms)``.
    """
    fit_kwargs = dict(fit_kwargs or {})
    structure = _state_to_structure(state)

    n_events = train.n_events
    LL0 = _null_loglik(train)
    # No ASCs in this DSL: constants-only MNL == uniform-over-choice-set.
    LLC = LL0

    row: Dict[str, Any] = {c: np.nan for c in _REWARD_COLUMNS}
    row['successfulEstimation'] = False
    row['LL0'] = LL0
    row['LLC'] = LLC
    row['numParams'] = int(len(structure.terms))

    try:
        feats_list = _features_for_structure(structure, train)
        weights = fit_weights(
            structure,
            feats_list,
            train.chosen_indices,
            train.customer_ids,
            train.categories,
            **fit_kwargs,
        )
        ll = _loglik_from_weights(
            feats_list, train.chosen_indices,
            train.customer_ids, train.categories, weights,
        )
    except Exception:
        return pd.DataFrame([row])

    if not np.isfinite(ll):
        return pd.DataFrame([row])

    num_params = _effective_n_params(structure, weights)

    # Validation LL: compute if val batch is provided (required for LLout
    # to be non-NaN; any reward_weights configuration that keys on LLout
    # must pass val through).
    if val is not None and val.n_events > 0:
        try:
            val_feats = _features_for_structure(structure, val)
            ll_out = _loglik_from_weights(
                val_feats, val.chosen_indices,
                val.customer_ids, val.categories, weights,
            )
        except Exception:
            ll_out = float('nan')
    else:
        ll_out = float('nan')

    aic = 2.0 * num_params - 2.0 * ll
    bic = float(np.log(max(n_events, 1))) * num_params - 2.0 * ll
    # rho-squared relative to the null (LL0) and constants-only (LLC) baselines.
    if LL0 != 0.0 and np.isfinite(LL0):
        rho2_0 = 1.0 - ll / LL0
        adj_rho2_0 = 1.0 - (ll - num_params) / LL0
    else:
        rho2_0 = float('nan')
        adj_rho2_0 = float('nan')
    if LLC != 0.0 and np.isfinite(LLC):
        rho2_C = 1.0 - ll / LLC
        adj_rho2_C = 1.0 - (ll - num_params) / LLC
    else:
        rho2_C = float('nan')
        adj_rho2_C = float('nan')

    row.update({
        'numParams': num_params,
        'successfulEstimation': True,
        'LL0': LL0,
        'LLC': LLC,
        'LLout': ll_out,
        'rho2_0': rho2_0,
        'adjRho2_0': adj_rho2_0,
        'rho2_C': rho2_C,
        'adjRho2_C': adj_rho2_C,
        'AIC': aic,
        'BIC': bic,
    })
    return pd.DataFrame([row])


# =============================================================================
# FittedBaseline + Baseline wrapper
# =============================================================================


@dataclass
class DelphosFitted:
    """Fitted Delphos baseline, conforming to the FittedBaseline protocol."""

    name: str
    best_structure: DSLStructure
    best_weights: HierarchicalWeights
    base_feature_names: List[str]
    train_nll: float
    val_nll: float
    n_episodes_run: int
    candidates_evaluated: int
    n_categories_fit: int = 0
    n_customers_fit: int = 0

    def score_events(self, batch: BaselineEventBatch) -> List[np.ndarray]:
        names = batch.base_feature_names
        scores: List[np.ndarray] = []
        for feats, cid, cat in zip(
            batch.base_features_list, batch.customer_ids, batch.categories
        ):
            struct_feats = build_structure_features(self.best_structure, feats, names)
            w = self.best_weights.get_weights(cid, cat)
            scores.append(struct_feats @ w)
        return scores

    @property
    def n_params(self) -> int:
        n_terms = len(self.best_structure.terms)
        return int(n_terms * (1 + self.n_categories_fit + self.n_customers_fit))

    @property
    def description(self) -> str:
        return (
            f"Delphos DQN {self.best_structure} "
            f"episodes={self.n_episodes_run} "
            f"evals={self.candidates_evaluated}"
        )


class Delphos:
    """DQN-based utility specification search (Arteaga, Paz et al.).

    Parameters
    ----------
    num_episodes : int
        Number of training episodes. Paper uses 50k; default here is 500
        to keep smoke runs tractable.
    hidden_layers : tuple of int
        Hidden layer sizes for the DQN.
    gamma : float
        Reward discount factor (also used for the discounted reward
        distribution across transitions within an episode).
    batch_size : int
        Experience-replay batch size.
    target_update_freq : int
        Episodes between hard copies policy_net -> target_net.
    reward_weights : dict, optional
        Delphos reward-metric weights. Default ``{'AIC': 1.0}`` (Delphos.py L41).
    reward_distribution_mode : str
        'exponential', 'linear', or 'uniform' — how to distribute the
        terminal reward across the episode's transitions.
    epsilon_min : float
        Lower bound on the epsilon-greedy exploration rate.
    min_percentage : float
        Fraction of num_episodes before early stopping is considered.
    early_stop_window : int
        Rolling window length for the Delphos early-stopping rule.
    patience : int
        Early-stopping patience (number of windows without improvement).
    fit_kwargs : dict, optional
        Passed through to inner_loop.fit_weights.
    seed : int
        RNG seed.
    """

    name = "Delphos"

    def __init__(
        self,
        num_episodes: int = 500,
        hidden_layers: Tuple[int, ...] = (128, 64),
        gamma: float = 0.9,
        batch_size: int = 64,
        target_update_freq: int = 10,
        reward_weights: Optional[Dict[str, float]] = None,
        reward_distribution_mode: str = 'exponential',
        epsilon_min: float = 0.01,
        min_percentage: float = 0.1,
        early_stop_window: int = 50,
        patience: int = 20,
        fit_kwargs: Optional[dict] = None,
        seed: int = 0,
    ):
        self.num_episodes = int(num_episodes)
        self.hidden_layers = tuple(hidden_layers)
        self.gamma = float(gamma)
        self.batch_size = int(batch_size)
        self.target_update_freq = int(target_update_freq)
        self.reward_weights = dict(reward_weights) if reward_weights else {'AIC': 1.0}
        self.reward_distribution_mode = reward_distribution_mode
        self.epsilon_min = float(epsilon_min)
        self.min_percentage = float(min_percentage)
        self.early_stop_window = int(early_stop_window)
        self.patience = int(patience)
        self.fit_kwargs = dict(fit_kwargs or {})
        self.seed = int(seed)
        # Set at fit time so tests can inspect the learner
        self.learner_: Optional[DQNLearner] = None

    def fit(
        self,
        train: BaselineEventBatch,
        val: BaselineEventBatch,
    ) -> DelphosFitted:
        # State space: ASC at slot 0, then |ALL_TERMS| feature slots.
        state_space_params = {
            'num_vars': len(ALL_TERMS) + 1,
            'transformations': ['linear', 'log', 'box-cox'],
            'taste': ['generic'],  # no 'specific' in list -> Delphos non-specific branch
            'covariates': [],
        }

        def estimator(state):
            return _estimate_from_state(
                state, train, val=val, fit_kwargs=self.fit_kwargs,
            )

        learner = DQNLearner(
            state_space_params=state_space_params,
            num_episodes=self.num_episodes,
            estimate_fn=estimator,
            hidden_layers=self.hidden_layers,
            discount_factor=self.gamma,
            batch_size=self.batch_size,
            target_update_freq=self.target_update_freq,
            patience=self.patience,
            early_stop_window=self.early_stop_window,
            min_percentage=self.min_percentage,
            reward_weights=self.reward_weights,
            reward_distribution=self.reward_distribution_mode,
            epsilon_min=self.epsilon_min,
            seed=self.seed,
        )
        self.learner_ = learner

        total_episodes = learner.train()

        # Pick the best candidate. Delphos's tracker stores best per metric.
        metric = learner.metric
        best_info = learner.best_candidates.get(metric, {})
        best_repr = best_info.get('representation')
        if best_repr is None:
            # Fallback: scan the cache for any successful estimation, pick
            # the one with the best AIC.
            best_repr = None
            best_val = np.inf
            for rep, df in learner._estimate_cache.items():
                if df.empty:
                    continue
                ok = bool(df['successfulEstimation'].iloc[0])
                if not ok:
                    continue
                aic = float(df['AIC'].iloc[0])
                if np.isfinite(aic) and aic < best_val:
                    best_val = aic
                    best_repr = rep
        if best_repr is None:
            # Final fallback: a trivial structure.
            best_state: List[Tuple] = []
        else:
            best_state = learner.state_manager.decode_string_to_state(best_repr)

        best_structure = _state_to_structure(best_state)

        # Re-fit hierarchical weights on train for the final artifact.
        train_feats = _features_for_structure(best_structure, train)
        best_weights = fit_weights(
            best_structure,
            train_feats,
            train.chosen_indices,
            train.customer_ids,
            train.categories,
            **self.fit_kwargs,
        )
        train_nll = -_loglik_from_weights(
            train_feats, train.chosen_indices,
            train.customer_ids, train.categories, best_weights,
        )
        val_feats = _features_for_structure(best_structure, val)
        val_nll = -_loglik_from_weights(
            val_feats, val.chosen_indices,
            val.customer_ids, val.categories, best_weights,
        )

        return DelphosFitted(
            name=self.name,
            best_structure=best_structure,
            best_weights=best_weights,
            base_feature_names=list(train.base_feature_names),
            train_nll=float(train_nll),
            val_nll=float(val_nll),
            n_episodes_run=int(total_episodes),
            candidates_evaluated=int(len(learner._estimate_cache)),
            n_categories_fit=len(set(train.categories)),
            n_customers_fit=len(set(train.customer_ids)),
        )
