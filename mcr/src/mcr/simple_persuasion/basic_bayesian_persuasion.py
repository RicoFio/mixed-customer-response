from dataclasses import dataclass, field
from typing import Any
import numpy as np


@dataclass
class BasicBayesianPersuasion:
    """
    Minimal Bayesian persuasion example (1 sender, 1 receiver, finite states/actions).

    State and action dimensions:
    - prior has shape (n_states,)
    - sender_utility and receiver_utility have shape (n_actions, n_states)
    - signaling scheme has shape (n_states, n_messages), where each row sums to 1

    Optimization strategy:
    - Parameterize signaling probabilities with per-state logits + row-wise softmax.
    - Use soft receiver best responses (logit/quantal response) for smooth updates.
    - Run Adam + finite-difference gradients until convergence.
    """

    prior: np.ndarray
    sender_utility: np.ndarray
    receiver_utility: np.ndarray
    n_messages: int = 2
    initial_signaling: np.ndarray | None = None
    temperature: float = 0.2
    seed: int = 0

    n_states: int = field(init=False)
    n_actions: int = field(init=False)
    _logits: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        self.prior = np.asarray(self.prior, dtype=float)
        self.sender_utility = np.asarray(self.sender_utility, dtype=float)
        self.receiver_utility = np.asarray(self.receiver_utility, dtype=float)

        if self.prior.ndim != 1 or self.prior.size < 2:
            raise ValueError("prior must be a 1D array with at least 2 states.")

        prior_sum = self.prior.sum()
        if prior_sum <= 0:
            raise ValueError("prior probabilities must sum to a positive value.")
        self.prior = self.prior / prior_sum

        if self.sender_utility.ndim != 2 or self.receiver_utility.ndim != 2:
            raise ValueError("Utility matrices must be 2D arrays (n_actions, n_states).")
        if self.sender_utility.shape != self.receiver_utility.shape:
            raise ValueError("sender_utility and receiver_utility must have the same shape.")
        if self.sender_utility.shape[1] != self.prior.size:
            raise ValueError("Utility matrices second dimension must match prior size.")
        if self.n_messages < 2:
            raise ValueError("n_messages must be at least 2.")
        if self.temperature <= 0:
            raise ValueError("temperature must be strictly positive.")

        self.n_actions, self.n_states = self.sender_utility.shape

        if self.initial_signaling is not None:
            init = np.asarray(self.initial_signaling, dtype=float)
            if init.shape != (self.n_states, self.n_messages):
                raise ValueError(
                    "initial_signaling must have shape (n_states, n_messages)."
                )
            if np.any(init < 0):
                raise ValueError("initial_signaling cannot contain negative values.")
            row_sums = init.sum(axis=1, keepdims=True)
            if np.any(row_sums <= 0):
                raise ValueError("Each initial_signaling row must sum to a positive value.")
            init = init / row_sums
            self._logits = np.log(np.clip(init, 1e-12, 1.0))
        else:
            rng = np.random.default_rng(self.seed)
            self._logits = rng.normal(
                loc=0.0, scale=0.05, size=(self.n_states, self.n_messages)
            )

    @staticmethod
    def _softmax(x: np.ndarray, axis: int) -> np.ndarray:
        shifted = x - np.max(x, axis=axis, keepdims=True)
        exp_shifted = np.exp(shifted)
        return exp_shifted / np.sum(exp_shifted, axis=axis, keepdims=True)

    def signaling_scheme(self, logits: np.ndarray | None = None) -> np.ndarray:
        """Return π(message | state) with shape (n_states, n_messages)."""
        if logits is None:
            logits = self._logits
        return self._softmax(logits, axis=1)

    def posterior_given_message(self, signaling: np.ndarray | None = None) -> np.ndarray:
        """
        Return posterior P(state | message) with shape (n_states, n_messages).
        """
        if signaling is None:
            signaling = self.signaling_scheme()

        joint = self.prior[:, None] * signaling
        message_prob = joint.sum(axis=0)

        posterior = np.zeros_like(joint)
        nonzero_messages = message_prob > 1e-12
        posterior[:, nonzero_messages] = (
            joint[:, nonzero_messages] / message_prob[None, nonzero_messages]
        )
        posterior[:, ~nonzero_messages] = self.prior[:, None]
        return posterior

    def receiver_policy(
        self, signaling: np.ndarray | None = None, hard_best_response: bool = False
    ) -> np.ndarray:
        """
        Return receiver policy P(action | message) with shape (n_actions, n_messages).
        """
        if signaling is None:
            signaling = self.signaling_scheme()

        posterior = self.posterior_given_message(signaling)
        expected_receiver_utility = self.receiver_utility @ posterior

        if hard_best_response:
            policy = np.zeros_like(expected_receiver_utility)
            best_actions = np.argmax(expected_receiver_utility, axis=0)
            policy[best_actions, np.arange(self.n_messages)] = 1.0
            return policy

        return self._softmax(expected_receiver_utility / self.temperature, axis=0)

    def expected_sender_utility(
        self, signaling: np.ndarray | None = None, hard_best_response: bool = False
    ) -> float:
        """Expected sender utility under prior, signaling and receiver response."""
        if signaling is None:
            signaling = self.signaling_scheme()

        posterior = self.posterior_given_message(signaling)
        receiver_policy = self.receiver_policy(
            signaling=signaling,
            hard_best_response=hard_best_response,
        )

        message_prob = (self.prior[:, None] * signaling).sum(axis=0)
        expected_sender_utility = self.sender_utility @ posterior
        sender_value_per_message = np.sum(
            receiver_policy * expected_sender_utility,
            axis=0,
        )
        return float(np.dot(message_prob, sender_value_per_message))

    def _objective_from_flat_logits(self, flat_logits: np.ndarray) -> float:
        logits = flat_logits.reshape(self.n_states, self.n_messages)
        signaling = self.signaling_scheme(logits=logits)
        return self.expected_sender_utility(signaling=signaling, hard_best_response=False)

    def _finite_difference_gradient(
        self, flat_logits: np.ndarray, epsilon: float
    ) -> np.ndarray:
        gradient = np.zeros_like(flat_logits)

        for i in range(flat_logits.size):
            direction = np.zeros_like(flat_logits)
            direction[i] = epsilon
            plus = self._objective_from_flat_logits(flat_logits + direction)
            minus = self._objective_from_flat_logits(flat_logits - direction)
            gradient[i] = (plus - minus) / (2.0 * epsilon)

        return gradient

    def solve(
        self,
        max_iter: int = 500,
        step_size: float = 0.15,
        finite_diff_epsilon: float = 1e-4,
        convergence_tol: float = 1e-8,
        convergence_patience: int = 30,
    ) -> dict[str, Any]:
        """
        Optimize the signaling scheme with Adam + finite differences.

        Returns a dictionary with convergence history and final policies.
        """
        if max_iter <= 0:
            raise ValueError("max_iter must be positive.")
        if step_size <= 0:
            raise ValueError("step_size must be positive.")
        if finite_diff_epsilon <= 0:
            raise ValueError("finite_diff_epsilon must be positive.")
        if convergence_patience <= 0:
            raise ValueError("convergence_patience must be positive.")

        flat_logits = self._logits.reshape(-1).copy()
        m = np.zeros_like(flat_logits)
        v = np.zeros_like(flat_logits)
        beta1 = 0.9
        beta2 = 0.999
        adam_eps = 1e-8

        utility_history: list[float] = []
        grad_norm_history: list[float] = []
        stagnant_steps = 0
        converged = False

        for t in range(1, max_iter + 1):
            utility = self._objective_from_flat_logits(flat_logits)
            grad = self._finite_difference_gradient(
                flat_logits=flat_logits,
                epsilon=finite_diff_epsilon,
            )
            grad_norm = float(np.linalg.norm(grad))

            m = beta1 * m + (1.0 - beta1) * grad
            v = beta2 * v + (1.0 - beta2) * (grad * grad)
            m_hat = m / (1.0 - beta1**t)
            v_hat = v / (1.0 - beta2**t)
            flat_logits = flat_logits + step_size * m_hat / (np.sqrt(v_hat) + adam_eps)

            utility_history.append(float(utility))
            grad_norm_history.append(grad_norm)

            if len(utility_history) >= 2:
                utility_delta = abs(utility_history[-1] - utility_history[-2])
                if utility_delta < convergence_tol:
                    stagnant_steps += 1
                else:
                    stagnant_steps = 0

            if stagnant_steps >= convergence_patience:
                converged = True
                break

        self._logits = flat_logits.reshape(self.n_states, self.n_messages)
        final_signaling = self.signaling_scheme()
        final_posteriors = self.posterior_given_message(final_signaling)
        final_soft_receiver_policy = self.receiver_policy(
            signaling=final_signaling,
            hard_best_response=False,
        )
        final_hard_receiver_policy = self.receiver_policy(
            signaling=final_signaling,
            hard_best_response=True,
        )

        return {
            "iterations": len(utility_history),
            "converged": converged,
            "utility_history": utility_history,
            "grad_norm_history": grad_norm_history,
            "final_signaling": final_signaling,
            "final_posteriors": final_posteriors,
            "final_soft_receiver_policy": final_soft_receiver_policy,
            "final_hard_receiver_policy": final_hard_receiver_policy,
            "final_sender_utility_soft": self.expected_sender_utility(
                signaling=final_signaling,
                hard_best_response=False,
            ),
            "final_sender_utility_hard": self.expected_sender_utility(
                signaling=final_signaling,
                hard_best_response=True,
            ),
        }

    @classmethod
    def simple_binary_example(cls, seed: int = 0) -> "BasicBayesianPersuasion":
        """
        Build a toy persuasion problem:
        - states: good product, bad product
        - actions: buy, do_not_buy
        - sender prefers 'buy' regardless of state
        - receiver buys only when posterior quality is high enough
        """
        prior = np.array([0.35, 0.65])  # P(good), P(bad)

        sender_utility = np.array(
            [
                [1.0, 1.0],  # buy
                [0.0, 0.0],  # do_not_buy
            ]
        )
        receiver_utility = np.array(
            [
                [1.0, -1.0],  # buy
                [0.0, 0.0],  # do_not_buy
            ]
        )

        initial_signaling = np.array(
            [
                [0.8, 0.2],  # P(m1|good), P(m2|good)
                [0.3, 0.7],  # P(m1|bad),  P(m2|bad)
            ]
        )

        return cls(
            prior=prior,
            sender_utility=sender_utility,
            receiver_utility=receiver_utility,
            n_messages=2,
            initial_signaling=initial_signaling,
            temperature=0.2,
            seed=seed,
        )
