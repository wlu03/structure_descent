"""DQN machinery vendored for the Delphos baseline.

Port of the DQN / replay buffer / state manager / training loop from
``old_pipeline/src/baselines/delphos.py`` rewritten in NumPy (the
upstream port used PyTorch; the new baseline suite is CPU-only and
ships without torch). The network is a 2-hidden-layer MLP trained with
MSE against a periodically-synced target copy via Adam.

Scope: Option B of the design doc. The state space has one slot per
``(variable, transformation_code)`` pair plus a leading ASC slot, so the
binary state vector length is ``1 + num_attrs * 4`` where
``num_attrs = num_vars - 1`` and the transformation codes are
``{0: none, 1: linear, 2: log, 3: box-cox}``. The ``taste`` and
``covariates`` knobs from the upstream implementation are NOT ported —
Delphos under Option B runs with ``taste=['generic']`` and
``covariates=[]`` only.
"""

from __future__ import annotations

import random
from collections import deque
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Columns the reward machinery reads from the estimator's single-row frame.
# ---------------------------------------------------------------------------
_REWARD_COLUMNS: Tuple[str, ...] = (
    "numParams",
    "successfulEstimation",
    "LL0",
    "LLC",
    "LLout",
    "rho2_0",
    "adjRho2_0",
    "rho2_C",
    "adjRho2_C",
    "AIC",
    "BIC",
)


# ---------------------------------------------------------------------------
# DQNetwork: plain MLP + Adam, implemented in NumPy.
# ---------------------------------------------------------------------------


class DQNetwork:
    """Fully-connected MLP with ReLU hidden activations.

    Kept intentionally small (default ``(128, 64)``) to mirror the paper
    and the upstream port. Forward pass and backprop are hand-rolled so
    we can avoid a torch dependency.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_layers: Tuple[int, ...] = (128, 64),
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.input_size = int(input_size)
        self.output_size = int(output_size)
        self.hidden_layers = tuple(int(h) for h in hidden_layers)
        self.rng = rng if rng is not None else np.random.default_rng(0)

        sizes = [self.input_size, *self.hidden_layers, self.output_size]
        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []
        for n_in, n_out in zip(sizes[:-1], sizes[1:]):
            # He init for ReLU MLP; small scale on the output layer keeps
            # early Q-values near zero so epsilon-greedy exploration
            # isn't biased by random network initialization.
            std = np.sqrt(2.0 / max(n_in, 1))
            self.weights.append(
                self.rng.normal(0.0, std, size=(n_in, n_out)).astype(np.float64)
            )
            self.biases.append(np.zeros((n_out,), dtype=np.float64))

    # ---- parameter management -------------------------------------------------

    def get_params(self) -> List[np.ndarray]:
        """Deep-copy snapshot of (W, b) pairs."""
        return [w.copy() for w in self.weights] + [b.copy() for b in self.biases]

    def set_params(self, params: List[np.ndarray]) -> None:
        """Restore parameters previously returned by :meth:`get_params`."""
        n = len(self.weights)
        for i in range(n):
            self.weights[i] = params[i].copy()
            self.biases[i] = params[n + i].copy()

    # ---- forward pass ---------------------------------------------------------

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass. ``x`` is ``(input_size,)`` or ``(batch, input_size)``."""
        single = x.ndim == 1
        h = x.reshape(1, -1) if single else x
        h = h.astype(np.float64, copy=False)
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            h = h @ w + b
            if i < len(self.weights) - 1:
                h = np.maximum(h, 0.0)  # ReLU
        return h[0] if single else h

    def _forward_with_cache(
        self, x: np.ndarray
    ) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]:
        """Forward pass that returns pre-activations and activations for backprop."""
        activations: List[np.ndarray] = [x.astype(np.float64, copy=False)]
        pre_activations: List[np.ndarray] = []
        h = activations[0]
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = h @ w + b
            pre_activations.append(z)
            if i < len(self.weights) - 1:
                h = np.maximum(z, 0.0)
            else:
                h = z
            activations.append(h)
        return h, activations, pre_activations

    def backward(
        self,
        x_batch: np.ndarray,
        action_batch: np.ndarray,
        target_batch: np.ndarray,
    ) -> Tuple[float, List[np.ndarray], List[np.ndarray]]:
        """Compute MSE gradient of Q(s, a) towards ``target_batch``.

        Only the selected action's Q value contributes per-sample; gradients
        through the non-selected outputs are zero. This matches the upstream
        ``gather(1, action_batch)`` semantics.
        """
        batch_size = x_batch.shape[0]
        q_all, activations, _ = self._forward_with_cache(x_batch)
        q_selected = q_all[np.arange(batch_size), action_batch]
        diff = q_selected - target_batch
        loss = float(np.mean(diff ** 2))

        # d loss / d q_all: zeros except at (i, action[i]).
        dL_dq = np.zeros_like(q_all)
        dL_dq[np.arange(batch_size), action_batch] = 2.0 * diff / batch_size

        grad_w: List[np.ndarray] = [None] * len(self.weights)
        grad_b: List[np.ndarray] = [None] * len(self.biases)
        delta = dL_dq
        for i in range(len(self.weights) - 1, -1, -1):
            a_prev = activations[i]
            grad_w[i] = a_prev.T @ delta
            grad_b[i] = delta.sum(axis=0)
            if i > 0:
                delta = delta @ self.weights[i].T
                # ReLU derivative at layer i-1.
                delta = delta * (activations[i] > 0.0)
        return loss, grad_w, grad_b


class AdamOptimizer:
    """Minimal Adam optimizer for the DQN parameter lists."""

    def __init__(
        self,
        shapes: List[Tuple[int, ...]],
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        self.lr = float(lr)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)
        self.m = [np.zeros(s, dtype=np.float64) for s in shapes]
        self.v = [np.zeros(s, dtype=np.float64) for s in shapes]
        self.t = 0

    def step(self, params: List[np.ndarray], grads: List[np.ndarray]) -> None:
        self.t += 1
        bc1 = 1.0 - self.beta1 ** self.t
        bc2 = 1.0 - self.beta2 ** self.t
        for i, (p, g) in enumerate(zip(params, grads)):
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * (g * g)
            m_hat = self.m[i] / bc1
            v_hat = self.v[i] / bc2
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


# ---------------------------------------------------------------------------
# ReplayBuffer.
# ---------------------------------------------------------------------------


class ReplayBuffer:
    """FIFO replay buffer of ``(state_vec, action_idx, reward, next_state_vec, done)``."""

    def __init__(self, max_size: int) -> None:
        self.buffer: deque = deque(maxlen=int(max_size))

    def add(self, transition: Tuple[Any, ...]) -> None:
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> Optional[List[Tuple[Any, ...]]]:
        if len(self.buffer) < batch_size:
            return None
        return random.sample(self.buffer, int(batch_size))

    def __len__(self) -> int:
        return len(self.buffer)


# ---------------------------------------------------------------------------
# StateManager (Option-B-only: taste='generic', covariates=[]).
# ---------------------------------------------------------------------------


class StateManager:
    """State <-> vector / string encoder for Delphos states.

    State representation: ``List[Tuple[var, trans]]`` where ``var == 0`` is
    the ASC slot with ``trans == 'linear'`` and ``var in [1, num_vars)``
    is a real feature with ``trans in {'linear', 'log', 'box-cox'}``.
    """

    _TRANSFORMATION_CODES: Dict[int, str] = {
        0: "none",
        1: "linear",
        2: "log",
        3: "box-cox",
    }

    def __init__(self, state_space_params: Dict[str, Any]) -> None:
        self.num_vars = int(state_space_params.get("num_vars", 1))
        self.transformations = list(
            state_space_params.get("transformations", ["linear", "log", "box-cox"])
        )
        # Option B: no 'specific' taste, no covariates.
        self.state_space_params = {
            "num_vars": self.num_vars,
            "transformations": self.transformations,
            "taste": ["generic"],
            "covariates": [],
        }
        self._inverse_mapping: Dict[str, int] = {
            v: k for k, v in self._TRANSFORMATION_CODES.items()
        }

    # ---- shape helpers --------------------------------------------------------

    def get_state_length(self) -> int:
        """1 (ASC slot) + num_attrs * 4 transformation-code slots."""
        num_attrs = max(self.num_vars - 1, 0)
        return 1 + num_attrs * 4

    # ---- encoders -------------------------------------------------------------

    def encode_state_to_vector(self, state: List[Tuple[int, str]]) -> np.ndarray:
        vec = np.zeros(self.get_state_length(), dtype=np.float64)
        asc_offset = 1
        num_transformations = 4
        for var, trans in state:
            if var == 0 and trans == "linear":
                vec[0] = 1.0
                continue
            if var < 1 or var >= self.num_vars:
                continue
            trans_idx = self._inverse_mapping.get(trans, 0)
            attr_idx = (var - 1) * num_transformations
            slot = asc_offset + attr_idx + trans_idx
            if 0 <= slot < vec.shape[0]:
                vec[slot] = 1.0
        return vec

    def encode_state_to_string(self, state: List[Tuple[int, str]]) -> str:
        representation = ["000"] * self.num_vars
        for entry in state:
            var, trans = entry[0], entry[1]
            if var < 0 or var >= self.num_vars:
                continue
            trans_code = self._inverse_mapping.get(trans, 0)
            representation[var] = f"{trans_code}00"
        return "_".join(representation)

    def decode_string_to_state(self, representation: str) -> List[Tuple[int, str]]:
        state: List[Tuple[int, str]] = []
        for idx, code in enumerate(representation.split("_")):
            if code == "000":
                continue
            trans_code = int(code[0])
            trans = self._TRANSFORMATION_CODES.get(trans_code, "none")
            state.append((idx, trans))
        return state

    # ---- legal-action masking ------------------------------------------------

    def mask_invalid_actions(
        self,
        state: List[Tuple[int, str]],
        action_space: List[Tuple[Any, ...]],
    ) -> List[Tuple[Any, ...]]:
        """Return the subset of ``action_space`` that is legal in ``state``.

        Rules (generic taste, no covariates):
        * ``terminate`` is always legal.
        * ASC (``var == 0``) can be added at most once per episode.
        * ``add(var, trans)`` is legal only if ``var`` is not already in
          the state.
        * ``change(var, trans)`` is legal only if the tuple
          ``(var, trans)`` is not already in the state.
        """
        add_current = {var for var, _ in state}
        change_current = {(var, trans) for var, trans in state}
        asc_added = 0 in add_current

        valid: List[Tuple[Any, ...]] = []
        seen_asc_action = False
        for action in action_space:
            act_type = action[0]
            if act_type == "terminate":
                valid.append(action)
                continue
            var = action[1]
            trans = action[2]
            if var == 0:
                if trans != "linear":
                    continue
                if asc_added or seen_asc_action:
                    continue
                valid.append(action)
                seen_asc_action = True
                continue
            if act_type == "add" and var not in add_current:
                valid.append(action)
            elif act_type == "change" and (var, trans) not in change_current:
                valid.append(action)
        return valid


# ---------------------------------------------------------------------------
# DQNLearner: full Delphos training loop (Option-B action space).
# ---------------------------------------------------------------------------


class DQNLearner:
    """DQN agent for Delphos-style utility-structure search.

    Only changes relative to the paper's ``DQNLearner``:
    * No disk IO (no ``rewards.csv``); the estimation cache is in-memory.
    * Torch is replaced with NumPy (2-hidden-layer MLP + Adam + MSE).
    * The user supplies ``estimate_fn(state) -> pd.DataFrame`` with the
      :data:`_REWARD_COLUMNS` schema; the learner never calls Apollo.
    """

    def __init__(
        self,
        state_space_params: Dict[str, Any],
        num_episodes: int,
        estimate_fn: Callable[[List[Tuple[int, str]]], pd.DataFrame],
        hidden_layers: Tuple[int, ...] = (128, 64),
        discount_factor: float = 0.9,
        learning_rate: float = 1e-3,
        buffer_size: int = 10000,
        batch_size: int = 64,
        target_update_freq: int = 10,
        patience: int = 20,
        early_stop_window: int = 50,
        early_stop_tolerance: float = 1e-3,
        min_percentage: float = 0.5,
        reward_weights: Optional[Dict[str, float]] = None,
        reward_distribution: str = "exponential",
        epsilon_min: float = 0.01,
        seed: int = 0,
    ) -> None:
        self.state_space_params = state_space_params
        self.num_episodes = int(num_episodes)
        self.discount_factor = float(discount_factor)
        self.batch_size = int(batch_size)
        self.target_update_freq = int(target_update_freq)
        self.epsilon = 1.0
        self.epsilon_min = float(epsilon_min)
        self.epsilon_decay = 1.0 / max(self.num_episodes, 1)
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.state_manager = StateManager(state_space_params)

        self.reward_weights = dict(reward_weights) if reward_weights else {"AIC": 1.0}
        default_metric_directions = {
            "AIC": "minimize",
            "BIC": "minimize",
            "adjRho2_0": "maximize",
            "rho2_0": "maximize",
            "rho2_C": "maximize",
            "adjRho2_C": "maximize",
            "LLout": "maximize",
        }
        self.metric_directions = {
            metric: default_metric_directions.get(metric, "maximize")
            for metric in self.reward_weights
        }
        self.metric = list(self.reward_weights.keys())[0]
        self.reward_distribution = reward_distribution

        self._estimate_cache: Dict[str, pd.DataFrame] = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self._estimate_fn = estimate_fn

        self.action_space, self.action_to_index = self._define_action_space()

        self.action_log: List[dict] = []
        self.training_log: List[dict] = []

        input_size = self.state_manager.get_state_length()
        output_size = len(self.action_space)

        # Deterministic RNGs for reproducible epsilon-greedy and network init.
        self._rng = np.random.default_rng(int(seed))
        random.seed(int(seed))
        np.random.seed(int(seed))

        self.policy_net = DQNetwork(
            input_size, output_size, hidden_layers, rng=self._rng
        )
        self.target_net = DQNetwork(
            input_size, output_size, hidden_layers, rng=self._rng
        )
        self.target_net.set_params(self.policy_net.get_params())

        param_shapes = [w.shape for w in self.policy_net.weights] + [
            b.shape for b in self.policy_net.biases
        ]
        self.optimizer = AdamOptimizer(param_shapes, lr=learning_rate)

        # Candidate tracking.
        self.best_candidates = {
            metric: {
                "value": (
                    -np.inf
                    if self.metric_directions[metric] == "maximize"
                    else np.inf
                ),
                "episode": -1,
                "representation": None,
            }
            for metric in self.reward_weights
        }
        self.LL0: Optional[float] = None
        self.current_best_history: List[dict] = []
        self.patience = int(patience)
        self.early_stop_window = int(early_stop_window)
        self.early_stop_tolerance = float(early_stop_tolerance)
        self.min_percentage = float(min_percentage)
        self.min_episodes_before_stop = self.num_episodes * self.min_percentage
        self.no_improvement_count = 0
        self.model_set: set = set()

    # ---- action space ---------------------------------------------------------

    def _define_action_space(
        self,
    ) -> Tuple[List[Tuple[Any, ...]], Dict[Tuple[Any, ...], int]]:
        actions: List[Tuple[Any, ...]] = [("terminate",)]
        num_vars = self.state_space_params["num_vars"]
        transformations = self.state_space_params["transformations"]
        for var in range(num_vars):
            if var == 0:
                actions.append(("add", var, "linear"))
                actions.append(("change", var, "linear"))
            else:
                for transformation in transformations:
                    actions.append(("add", var, transformation))
                    actions.append(("change", var, transformation))
        action_to_index = {action: idx for idx, action in enumerate(actions)}
        return actions, action_to_index

    # ---- action selection ----------------------------------------------------

    def select_action_index(self, state: List[Tuple[int, str]]) -> int:
        valid_actions = self.state_manager.mask_invalid_actions(
            state, self.action_space
        )
        valid_indices = [self.action_to_index[a] for a in valid_actions]
        if np.random.rand() < self.epsilon:
            return int(random.choice(valid_indices))
        state_vec = self.state_manager.encode_state_to_vector(state)
        q_values = self.policy_net.forward(state_vec)
        masked = np.full_like(q_values, -np.inf)
        masked[valid_indices] = q_values[valid_indices]
        return int(np.argmax(masked))

    def apply_action(
        self, state: List[Tuple[int, str]], action_index: int
    ) -> Tuple[List[Tuple[int, str]], bool]:
        state = list(state)
        action = self.action_space[action_index]
        if action[0] == "terminate":
            return state, True
        _, var, trans = action
        if action[0] == "add":
            state.append((var, trans))
        elif action[0] == "change":
            state = [(v, t) if v != var else (v, trans) for v, t in state]
        return state, False

    # ---- episode generation --------------------------------------------------

    def generate_episode(
        self, episode_count: int
    ) -> Tuple[List[Tuple[Any, ...]], List[Tuple[int, str]], bool]:
        state: List[Tuple[int, str]] = []
        episode_steps: List[Tuple[Any, ...]] = []
        done = False
        while not done:
            action_index = self.select_action_index(state)
            next_state, done = self.apply_action(state, action_index)
            episode_steps.append((state, action_index, next_state))
            self.action_log.append(
                {
                    "episode": episode_count,
                    "state": state,
                    "action": self.action_space[action_index],
                    "next_state": next_state,
                }
            )
            state = next_state
        return episode_steps, state, done

    # ---- candidate tracking / early stopping ---------------------------------

    def update_candidate_tracker(
        self,
        episode_count: int,
        modelling_outcomes: Dict[str, float],
        state: List[Tuple[int, str]],
    ) -> None:
        best_repr = self.state_manager.encode_state_to_string(state)
        for metric in self.reward_weights:
            if metric not in modelling_outcomes:
                continue
            value = modelling_outcomes[metric]
            if not isinstance(value, (int, float, np.integer, np.floating)):
                continue
            if pd.isna(value) or not np.isfinite(value):
                continue
            current_best = self.best_candidates[metric]
            direction = self.metric_directions.get(metric, "maximize")
            improved = (
                value > current_best["value"]
                if direction == "maximize"
                else value < current_best["value"]
            )
            if improved:
                self.best_candidates[metric] = {
                    "value": float(value),
                    "episode": int(episode_count),
                    "representation": best_repr,
                }
        snapshot = {"episode": int(episode_count)}
        for metric, info in self.best_candidates.items():
            snapshot[f"{metric}_value"] = info["value"]
            snapshot[f"{metric}_repr"] = info["representation"]
        self.current_best_history.append(snapshot)

    def check_early_stopping(self, episode_rewards: List[float]) -> bool:
        if len(episode_rewards) < 2 * self.early_stop_window:
            return False
        current_window = episode_rewards[-self.early_stop_window :]
        previous_window = episode_rewards[
            -2 * self.early_stop_window : -self.early_stop_window
        ]
        current_mean = float(np.mean(current_window))
        previous_mean = float(np.mean(previous_window))
        relative_improvement = (current_mean - previous_mean) / (
            abs(previous_mean) + 1e-8
        )
        if relative_improvement < self.early_stop_tolerance:
            self.no_improvement_count += 1
        else:
            self.no_improvement_count = 0
        return self.no_improvement_count >= self.patience

    # ---- reward normalization + function -------------------------------------

    def normalize_reward_metric(self, metric: str, value: float) -> float:
        values = [
            log[metric]
            for log in self.training_log
            if metric in log and pd.notna(log[metric])
        ]
        max_val = max(values) if values else 0.0
        min_val = min(values) if values else 0.0
        if not values or self.LL0 is None:
            return 0.0
        norm = 0.0
        if metric == "LLout":
            if value <= self.LL0 or max_val - self.LL0 == 0:
                return 0.0
            norm = (value - self.LL0) / (max_val - self.LL0)
        elif metric in ("AIC", "BIC"):
            max_value = -2 * self.LL0
            if value >= max_value or max_val == min_val:
                return 0.0
            norm = (max_val - value) / (max_val - min_val)
        elif metric in ("rho2_0", "adjRho2_0", "rho2_C", "adjRho2_C"):
            if value <= 0:
                return 0.0
            norm = float(value)
        if np.isfinite(norm) and not pd.isna(norm):
            return float(norm)
        return 0.0

    def reward_function(self, modelling_outcome: pd.DataFrame) -> float:
        try:
            if (
                modelling_outcome.empty
                or bool(modelling_outcome["successfulEstimation"].values[0]) is False
            ):
                return 0.0
            reward = 0.0
            for metric, weight in self.reward_weights.items():
                if metric in modelling_outcome.columns:
                    raw_value = modelling_outcome[metric].values[0]
                    if pd.isna(raw_value):
                        normalized_value = 0.0
                    else:
                        normalized_value = self.normalize_reward_metric(
                            metric, float(raw_value)
                        )
                    reward += weight * normalized_value
            return float(reward)
        except Exception:
            return 0.0

    # ---- cached estimator call ----------------------------------------------

    def delphos_interaction(
        self, state: List[Tuple[int, str]]
    ) -> Tuple[float, pd.DataFrame]:
        key = self.state_manager.encode_state_to_string(state)
        if key in self._estimate_cache:
            self.cache_hits += 1
            modelling_outcomes = self._estimate_cache[key]
        else:
            self.cache_misses += 1
            modelling_outcomes = self._estimate_fn(state)
            self._estimate_cache[key] = modelling_outcomes

        if self.LL0 is None and "LL0" in modelling_outcomes.columns:
            ll0_val = modelling_outcomes["LL0"].values[0]
            if pd.notna(ll0_val) and np.isfinite(ll0_val):
                self.LL0 = float(ll0_val)

        final_reward = self.reward_function(modelling_outcomes)
        keep_cols = set(self.reward_weights.keys()) | {
            c for c in _REWARD_COLUMNS if c in modelling_outcomes.columns
        }
        reduced = modelling_outcomes[
            modelling_outcomes.columns.intersection(keep_cols)
        ]
        return final_reward, reduced

    # ---- experience replay --------------------------------------------------

    def perform_experience_replay(self) -> None:
        batch = self.replay_buffer.sample(self.batch_size)
        if not batch:
            return
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(
            *batch
        )
        state_arr = np.stack(state_batch).astype(np.float64, copy=False)
        action_arr = np.asarray(action_batch, dtype=np.int64)
        reward_arr = np.asarray(reward_batch, dtype=np.float64)
        next_state_arr = np.stack(next_state_batch).astype(np.float64, copy=False)
        done_arr = np.asarray(done_batch, dtype=bool)

        next_q = self.target_net.forward(next_state_arr)
        max_next_q = next_q.max(axis=1)
        target = reward_arr + (1.0 - done_arr.astype(np.float64)) * (
            self.discount_factor * max_next_q
        )

        _, grad_w, grad_b = self.policy_net.backward(state_arr, action_arr, target)
        params = self.policy_net.weights + self.policy_net.biases
        grads = list(grad_w) + list(grad_b)
        self.optimizer.step(params, grads)

    # ---- main loop ----------------------------------------------------------

    def train(self) -> int:
        """Run the DQN training loop. Returns number of episodes actually run."""
        episode_count = 0
        episode_rewards: List[float] = []
        self.current_best_history = []

        for episode_count in range(self.num_episodes):
            episode_steps, state, _ = self.generate_episode(episode_count)
            final_reward, modelling_outcomes = self.delphos_interaction(state)
            episode_rewards.append(final_reward)
            final_repr = self.state_manager.encode_state_to_string(state)
            self.model_set.add(final_repr)

            L = len(episode_steps)
            gamma = self.discount_factor

            if self.reward_distribution == "uniform":
                inter_reward = final_reward / L if L > 0 else final_reward
                for step_state, step_action, step_next_state in episode_steps:
                    self.replay_buffer.add(
                        (
                            self.state_manager.encode_state_to_vector(step_state),
                            step_action,
                            inter_reward,
                            self.state_manager.encode_state_to_vector(step_next_state),
                            True,
                        )
                    )
            elif self.reward_distribution == "linear":
                for l, (step_state, step_action, step_next_state) in enumerate(
                    episode_steps
                ):
                    inter_reward = (
                        (final_reward * (l + 1)) / L if L > 0 else final_reward
                    )
                    self.replay_buffer.add(
                        (
                            self.state_manager.encode_state_to_vector(step_state),
                            step_action,
                            inter_reward,
                            self.state_manager.encode_state_to_vector(step_next_state),
                            True,
                        )
                    )
            else:  # 'exponential'
                for l, (step_state, step_action, step_next_state) in enumerate(
                    episode_steps
                ):
                    discounted = (
                        (gamma ** (L - l - 1)) * final_reward if L > 0 else final_reward
                    )
                    self.replay_buffer.add(
                        (
                            self.state_manager.encode_state_to_vector(step_state),
                            step_action,
                            discounted,
                            self.state_manager.encode_state_to_vector(step_next_state),
                            True,
                        )
                    )

            log: Dict[str, Any] = {
                "episode": int(episode_count),
                "specification": final_repr,
                "reward": float(final_reward),
                "epsilon": float(self.epsilon),
            }
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
                self.target_net.set_params(self.policy_net.get_params())

            self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_decay)

            if episode_count >= self.min_episodes_before_stop:
                if self.check_early_stopping(episode_rewards):
                    break

        return int(episode_count + 1)


__all__ = [
    "DQNLearner",
    "DQNetwork",
    "ReplayBuffer",
    "StateManager",
    "_REWARD_COLUMNS",
]
