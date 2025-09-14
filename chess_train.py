#!/usr/bin/env python3
# Copyright 2023 The Pgx Authors. All Rights Reserved.

import datetime
import os
import pickle
import time
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import mctx
import optax
import pgx
import wandb
from pgx.experimental import auto_reset
import haiku as hk
from omegaconf import OmegaConf
from pydantic import BaseModel

# Set JAX to use BF16 precision
# JAX only supports 32 or 64 for default dtype bits, so we'll use explicit casting instead
jax.config.update("jax_enable_x64", False)
jax.config.update("jax_default_matmul_precision", 'bfloat16')

# Define custom dtype for BF16 - always use JAX's bfloat16
BF16 = jnp.bfloat16

class Config(BaseModel):
    env_id: str = "chess"  # Using chess environment
    seed: int = 42
    max_num_iters: int = 300
    # network params
    num_channels: int = 128
    num_layers: int = 6  # Deeper network for chess
    resnet_v2: bool = True
    # selfplay params
    selfplay_batch_size: int = 256
    num_simulations: int = 64  
    max_num_steps: int = 256  
    # training params
    training_batch_size: int = 4096
    learning_rate: float = 0.0002
    # eval params
    eval_interval: int = 1
    # precision params
    use_bf16: bool = True  # Enable BF16 precision
    # baseline model
    baseline: str = "./baseline.ckpt"

    class Config:
        extra = "forbid"


# Only parse command line arguments when running the script directly
if __name__ == "__main__":
    # Parse command line arguments
    conf_dict = OmegaConf.from_cli()
    config: Config = Config(**conf_dict)
    print(config)
else:
    # Default config when imported
    config = Config()

# Create chess environment
env = pgx.make(config.env_id)

def random_policy(observation):
    """Simple random policy for evaluation."""
    batch_size = observation.shape[0]
    logits = jnp.zeros((batch_size, env.num_actions))
    return logits, jnp.zeros(batch_size)

# Check available devices
devices = jax.local_devices()
num_devices = len(devices)
if __name__ == "__main__":
    print(f"Number of devices: {num_devices}")


class AZNet(hk.Module):
    """AlphaZero neural network for chess."""

    def __init__(
        self,
        num_actions,
        num_channels: int = 128,
        num_blocks: int = 10,
        resnet_v2: bool = True,
        name="chess_az_net",
    ):
        super().__init__(name=name)
        self.num_actions = num_actions
        self.num_channels = num_channels
        self.num_blocks = num_blocks
        self.resnet_v2 = resnet_v2
        
    def __call__(self, x, is_training, test_local_stats):
        # Initial convolutional layer
        x = hk.Conv2D(self.num_channels, kernel_shape=3, padding="SAME")(x)
        
        if not self.resnet_v2:
            x = hk.BatchNorm(True, True, 0.9)(x, is_training, test_local_stats)
            x = jax.nn.relu(x)
        
        # Residual blocks
        for i in range(self.num_blocks):
            residual = x
            
            if self.resnet_v2:
                x = hk.BatchNorm(True, True, 0.9)(x, is_training, test_local_stats)
                x = jax.nn.relu(x)
            
            x = hk.Conv2D(self.num_channels, kernel_shape=3, padding="SAME")(x)
            x = hk.BatchNorm(True, True, 0.9)(x, is_training, test_local_stats)
            x = jax.nn.relu(x)
            x = hk.Conv2D(self.num_channels, kernel_shape=3, padding="SAME")(x)
            
            if self.resnet_v2:
                x = x + residual
            else:
                x = hk.BatchNorm(True, True, 0.9)(x, is_training, test_local_stats)
                x = jax.nn.relu(x + residual)
        
        if self.resnet_v2:
            x = hk.BatchNorm(True, True, 0.9)(x, is_training, test_local_stats)
            x = jax.nn.relu(x)
        
        # Policy head
        policy = hk.Conv2D(output_channels=32, kernel_shape=1)(x)
        policy = hk.BatchNorm(True, True, 0.9)(policy, is_training, test_local_stats)
        policy = jax.nn.relu(policy)
        policy = hk.Flatten()(policy)
        policy = hk.Linear(self.num_actions)(policy)
        
        # Value head
        value = hk.Conv2D(output_channels=8, kernel_shape=1)(x)
        value = hk.BatchNorm(True, True, 0.9)(value, is_training, test_local_stats)
        value = jax.nn.relu(value)
        value = hk.Flatten()(value)
        value = hk.Linear(self.num_channels)(value)
        value = jax.nn.relu(value)
        value = hk.Linear(1)(value)
        value = jnp.tanh(value)
        value = value.reshape((-1,))
        
        return policy, value


# Mixed precision is handled by JAX's jax_default_matmul_precision='bfloat16'
# This provides BF16 computation without problematic explicit casting
print(f"Using JAX matmul precision: {jax.config.read('jax_default_matmul_precision')} for efficient TPU computation")


def forward_fn(x, is_eval=False):
    """Forward function with safe dtypes: inputs -> FP32, compute precision set by JAX config."""
    net = AZNet(
        num_actions=env.num_actions,
        num_channels=config.num_channels,
        num_blocks=config.num_layers,
        resnet_v2=config.resnet_v2,
    )
    # Keep inputs in stable FP32 - JAX matmul precision config handles BF16 computation
    x = jnp.asarray(x, dtype=jnp.float32)
    policy_out, value_out = net(x, is_training=not is_eval, test_local_stats=False)
    return policy_out, value_out


# Transform forward function and create optimizer
forward = hk.without_apply_rng(hk.transform_with_state(forward_fn))
# Create optimizer - Adam with standard settings
optimizer = optax.adam(learning_rate=config.learning_rate)


def recurrent_fn(model, rng_key: jnp.ndarray, action: jnp.ndarray, state: pgx.State):
    """Recurrent function for MCTS."""
    del rng_key
    model_params, model_state = model

    # Store current player before step
    current_player = state.current_player
    # Apply action to environment
    state = jax.vmap(env.step)(state, action)

    # Get policy and value from model
    (logits, value), _ = forward.apply(model_params, model_state, state.observation, is_eval=True)
    
    # Mask invalid actions
    logits = logits - jnp.max(logits, axis=-1, keepdims=True)
    logits = jnp.where(state.legal_action_mask, logits, jnp.finfo(jnp.float32).min)

    # Get rewards for current player and calculate discount
    reward = state.rewards[jnp.arange(state.rewards.shape[0]), current_player]
    discount = -jnp.ones_like(value)
    
    # Set value to 0 if terminal state
    value = jnp.where(state.terminated, jnp.zeros_like(value), value)
    discount = jnp.where(state.terminated, jnp.zeros_like(discount), discount)
    
    # Keep values in FP32 for stability
    reward = reward.astype(jnp.float32)
    discount = discount.astype(jnp.float32)
    value = value.astype(jnp.float32)

    # Return output for MCTS
    recurrent_fn_output = mctx.RecurrentFnOutput(
        reward=reward,
        discount=discount,
        prior_logits=logits,
        value=value,
    )
    
    return recurrent_fn_output, state


class SelfplayOutput(NamedTuple):
    """Output from selfplay."""
    obs: jnp.ndarray
    reward: jnp.ndarray
    terminated: jnp.ndarray
    action_weights: jnp.ndarray
    discount: jnp.ndarray


@jax.pmap
def selfplay(model, rng_key: jnp.ndarray) -> SelfplayOutput:
    """Run selfplay with model."""
    model_params, model_state = model
    batch_size = config.selfplay_batch_size // num_devices

    def step_fn(state, key) -> SelfplayOutput:
        key1, key2 = jax.random.split(key)
        observation = state.observation

        # Get policy and value from model (forward_fn handles dtype conversion)
        (logits, value), _ = forward.apply(
            model_params, model_state, observation, is_eval=True
        )
        
        # Create root for MCTS
        root = mctx.RootFnOutput(prior_logits=logits, value=value, embedding=state)

        # Run MCTS
        policy_output = mctx.gumbel_muzero_policy(
            params=model,
            rng_key=key1,
            root=root,
            recurrent_fn=recurrent_fn,
            num_simulations=config.num_simulations,
            invalid_actions=~state.legal_action_mask,
            qtransform=mctx.qtransform_completed_by_mix_value,
            gumbel_scale=jnp.array(1.0, dtype=jnp.float32),
        )
        
        # Record current player
        actor = state.current_player
        
        # Apply action and auto-reset if terminated
        keys = jax.random.split(key2, batch_size)
        state = jax.vmap(auto_reset(env.step, env.init))(state, policy_output.action, keys)
        
        # Calculate discount in FP32
        discount = -jnp.ones_like(value, dtype=jnp.float32)
        discount = jnp.where(state.terminated, jnp.zeros_like(discount), discount)
        
        # Return new state and output
        # Return output (keep in native dtype, will be converted later if needed)
        return state, SelfplayOutput(
            obs=observation,
            action_weights=policy_output.action_weights,
            reward=state.rewards[jnp.arange(state.rewards.shape[0]), actor],
            terminated=state.terminated,
            discount=discount,
        )

    # Run selfplay for max_num_steps
    rng_key, sub_key = jax.random.split(rng_key)
    keys = jax.random.split(sub_key, batch_size)
    state = jax.vmap(env.init)(keys)
    key_seq = jax.random.split(rng_key, config.max_num_steps)
    _, data = jax.lax.scan(step_fn, state, key_seq)

    return data


class Sample(NamedTuple):
    """Training sample."""
    obs: jnp.ndarray
    policy_tgt: jnp.ndarray
    value_tgt: jnp.ndarray
    mask: jnp.ndarray


@jax.pmap
def compute_loss_input(data: SelfplayOutput) -> Sample:
    """Compute training targets from selfplay data."""
    batch_size = config.selfplay_batch_size // num_devices
    
    # Compute mask for value loss (1 if we have value target)
    value_mask = jnp.cumsum(data.terminated[::-1, :], axis=0)[::-1, :] >= 1

    # Compute value target by backwards pass
    def body_fn(carry, i):
        ix = config.max_num_steps - i - 1
        # Use FP32 for value computation
        reward = data.reward[ix].astype(jnp.float32)
        discount = data.discount[ix].astype(jnp.float32)
        v = reward + discount * carry
        return v, v

    _, value_tgt = jax.lax.scan(
        body_fn,
        jnp.zeros(batch_size, dtype=jnp.float32),
        jnp.arange(config.max_num_steps),
    )
    # Reverse to get targets in forward order
    value_tgt = value_tgt[::-1, :]

    # Keep targets in FP32 for numerical stability
    return Sample(
        obs=data.obs,
        policy_tgt=data.action_weights.astype(jnp.float32),
        value_tgt=value_tgt.astype(jnp.float32),
        mask=value_mask,
    )


def loss_fn(model_params, model_state, samples: Sample):
    """Compute loss for training."""
    (logits, value), model_state = forward.apply(
        model_params, model_state, samples.obs, is_eval=False
    )

    # Loss computation in FP32 for numerical stability
    logits = logits.astype(jnp.float32)
    value = value.astype(jnp.float32)
    policy_tgt = samples.policy_tgt.astype(jnp.float32)
    value_tgt = samples.value_tgt.astype(jnp.float32)

    # Policy loss (cross entropy)
    policy_loss = optax.softmax_cross_entropy(logits, policy_tgt)
    policy_loss = jnp.mean(policy_loss)

    # Value loss (L2)
    value_loss = optax.l2_loss(value, value_tgt)
    value_loss = jnp.mean(value_loss * samples.mask)  # masked for truncated episodes

    # Combined loss
    return policy_loss + value_loss, (model_state, policy_loss, value_loss)


@partial(jax.pmap, axis_name="i")
def train(model, opt_state, data: Sample):
    """Training function."""
    model_params, model_state = model
    
    # Compute gradients
    grads, (model_state, policy_loss, value_loss) = jax.grad(loss_fn, has_aux=True)(
        model_params, model_state, data
    )
    
    # Average gradients across devices
    grads = jax.lax.pmean(grads, axis_name="i")
    
    # Update model
    updates, opt_state = optimizer.update(grads, opt_state)
    model_params = optax.apply_updates(model_params, updates)
    model = (model_params, model_state)
    
    return model, opt_state, policy_loss, value_loss


# Note: We keep parameters in FP32 for numerical stability
# Only activations/computations use BF16 when configured


def ensure_jax_arrays(tree):
    """Convert all numpy arrays in a tree to JAX arrays (always to FP32 for storage)."""
    import numpy as np
    def _convert(x):
        # If it's a numpy array, convert to JAX array
        if isinstance(getattr(x, "__array__", None), np.ndarray) or (hasattr(x, "dtype") and not isinstance(x, jnp.ndarray)):
            try:
                # All float types (including bfloat16) -> float32 for storage
                if str(x.dtype).startswith(("float", "bfloat16")):
                    return jnp.asarray(x, dtype=jnp.float32)
                else:
                    return jnp.asarray(x)
            except Exception:
                return jnp.asarray(x)
        return x
    return jax.tree_util.tree_map(_convert, tree)

def load_model_from_checkpoint(checkpoint_path):
    """Load model from checkpoint file."""
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint {checkpoint_path} not found.")
        return None
    
    print(f"Loading model from {checkpoint_path}...")
    try:
        with open(checkpoint_path, "rb") as f:
            checkpoint = pickle.load(f)
        
        # Ensure all arrays are JAX arrays (not numpy)
        if "model" in checkpoint:
            checkpoint["model"] = ensure_jax_arrays(checkpoint["model"])
        if "opt_state" in checkpoint:
            checkpoint["opt_state"] = ensure_jax_arrays(checkpoint["opt_state"])
        
        # Don't cast model parameters to BF16 - keep them in FP32 for storage
            
        return checkpoint
    except (pickle.UnpicklingError, EOFError) as e:
        print(f"Error loading checkpoint: {e}")
        print(f"The checkpoint file {checkpoint_path} appears to be corrupted or truncated.")
        return None


@jax.pmap
def evaluate(rng_key, my_model, baseline_model):
    """Evaluate model against baseline."""
    my_player = 0
    my_model_params, my_model_state = my_model
    
    key, subkey = jax.random.split(rng_key)
    batch_size = config.selfplay_batch_size // num_devices
    keys = jax.random.split(subkey, batch_size)
    state = jax.vmap(env.init)(keys)

    def body_fn(val):
        key, state, R = val
        
        # Get logits from my model - forward_fn handles dtype conversion
        observation = state.observation
        (my_logits, _), _ = forward.apply(
            my_model_params, my_model_state, observation, is_eval=True
        )
        
        # Get logits from baseline model or use random policy
        if baseline_model is not None:
            baseline_params, baseline_state = baseline_model
            if baseline_state is None:
                baseline_state = {}
            (opp_logits, _), _ = forward.apply(
                baseline_params, baseline_state, observation, is_eval=True
            )
        else:
            opp_logits, _ = random_policy(observation)
        
        # Use my model when my turn, else use baseline/random policy
        is_my_turn = (state.current_player == my_player).reshape((-1, 1))
        logits = jnp.where(is_my_turn, my_logits, opp_logits)
        
        # Sample action
        key, subkey = jax.random.split(key)
        action = jax.random.categorical(subkey, logits, axis=-1)
        
        # Apply action
        state = jax.vmap(env.step)(state, action)
        
        # Accumulate rewards
        R = R + state.rewards[jnp.arange(batch_size), my_player]
        
        return (key, state, R)

    # Run until all games terminated
    _, _, R = jax.lax.while_loop(
        lambda x: ~(x[1].terminated.all()), body_fn, (key, state, jnp.zeros(batch_size))
    )
    
    return R


if __name__ == "__main__":
    # Initialize wandb
    wandb.init(project="pgx-chess-az", config=config.model_dump())
    
    # Log mixed precision usage
    wandb.config.update({"using_mixed_precision": config.use_bf16})

    # Initialize model and optimizer
    print("Initializing model...")
    dummy_state = jax.vmap(env.init)(jax.random.split(jax.random.PRNGKey(0), 2))
    # Always use FP32 for initialization
    dummy_input = dummy_state.observation.astype(jnp.float32)
    
    # Always try to load from baseline for initial model
    checkpoint = load_model_from_checkpoint(config.baseline)
    baseline_model = None  # For evaluation
    if checkpoint is not None:
        print(f"Loading initial model from baseline: {config.baseline}")
        model = checkpoint["model"]
        # Keep model parameters in FP32 for storage (compute will use BF16 if configured)
        model_params, model_state = model
        # Store the baseline model for evaluation
        baseline_model = model
        
        # Initialize new optimizer state for the loaded model
        # Convert params to FP32 for optimizer initialization to avoid BF16 dtype issues
        params_fp32 = jax.tree_util.tree_map(
            lambda x: x.astype(jnp.float32) if hasattr(x, 'dtype') else x, 
            model[0]
        )
        opt_state = optimizer.init(params=params_fp32)
        
        # Continue training from checkpoint's iteration and stats if available
        if "iteration" in checkpoint:
            iteration = checkpoint["iteration"]
            print(f"Continuing from iteration {iteration}")
        else:
            iteration = 0
            
        if "hours" in checkpoint:
            hours = checkpoint["hours"]
            print(f"Continuing from {hours:.2f} training hours")
        else:
            hours = 0.0
            
        if "frames" in checkpoint:
            frames = checkpoint["frames"]
            print(f"Continuing from {frames} frames")
        else:
            frames = 0
    else:
        print("Warning: Baseline checkpoint not found. Initializing new model.")
        # Initialize with FP32 input
        model = forward.init(jax.random.PRNGKey(0), dummy_input)
        # Keep model parameters in FP32 for storage
        model_params, model_state = model
        # Convert params to FP32 for optimizer initialization to avoid BF16 dtype issues
        params_fp32 = jax.tree_util.tree_map(
            lambda x: x.astype(jnp.float32) if hasattr(x, 'dtype') else x,
            model[0]
        )
        opt_state = optimizer.init(params=params_fp32)
        iteration = 0
        hours = 0.0
        frames = 0
    
    # Put model and optimizer on devices
    model, opt_state = jax.device_put_replicated((model, opt_state), devices)
    
    # Replicate baseline model if available
    baseline_model_replicated = None
    if baseline_model is not None:
        baseline_model_replicated = jax.device_put_replicated(baseline_model, devices)
    else:
        print("Warning: No baseline model loaded. Using random policy for evaluation.")
        baseline_model_replicated = jax.device_put_replicated(None, devices)

    # Log device type and precision being used
    device_type = jax.devices()[0].platform
    precision = "BF16" if config.use_bf16 else "FP32"
    print(f"Running on {device_type} devices with {precision} precision")
    wandb.config.update({"device_type": device_type, "precision": precision})

    # Prepare checkpoint directory
    now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9)))
    now = now.strftime("%Y%m%d%H%M%S")
    ckpt_dir = os.path.join("checkpoints", f"{config.env_id}_{now}")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Initialize logging
    log = {"iteration": iteration, "hours": hours, "frames": frames}

    # Main training loop
    rng_key = jax.random.PRNGKey(config.seed)
    print("Starting training...")
    
    while True:
        if iteration % config.eval_interval == 0:
            print(f"Evaluating at iteration {iteration}...")
            # Evaluate against baseline
            rng_key, subkey = jax.random.split(rng_key)
            keys = jax.random.split(subkey, num_devices)
            R = evaluate(keys, model, baseline_model_replicated)
            log.update(
                {
                    f"eval/vs_baseline/avg_R": R.mean().item(),
                    f"eval/vs_baseline/win_rate": ((R == 1).sum() / R.size).item(),
                    f"eval/vs_baseline/draw_rate": ((R == 0).sum() / R.size).item(),
                    f"eval/vs_baseline/lose_rate": ((R == -1).sum() / R.size).item(),
                }
            )

            # Save checkpoint
            print(f"Saving checkpoint at iteration {iteration}...")
            model_0, opt_state_0 = jax.tree_util.tree_map(lambda x: x[0], (model, opt_state))
            # Convert model to CPU and ensure it's in a serializable format
            model_cpu = jax.device_get(model_0)
            opt_state_cpu = jax.device_get(opt_state_0)
            with open(os.path.join(ckpt_dir, f"{iteration:06d}.ckpt"), "wb") as f:
                dic = {
                    "config": config,
                    "rng_key": rng_key,
                    "model": model_cpu,
                    "opt_state": opt_state_cpu,
                    "iteration": iteration,
                    "frames": frames,
                    "hours": hours,
                    "pgx.__version__": pgx.__version__,
                    "env_id": env.id,
                    "env_version": env.version,
                }
                pickle.dump(dic, f)

        print(log)
        wandb.log(log)

        if iteration >= config.max_num_iters:
            print(f"Training completed after {iteration} iterations.")
            break

        iteration += 1
        log = {"iteration": iteration}
        st = time.time()

        # Run selfplay
        print(f"Running selfplay for iteration {iteration}...")
        rng_key, subkey = jax.random.split(rng_key)
        keys = jax.random.split(subkey, num_devices)
        data = selfplay(model, keys)
        samples = compute_loss_input(data)

        # Prepare training data
        print("Preparing training data...")
        samples = jax.device_get(samples)  # (#devices, batch, max_num_steps, ...)
        frames += samples.obs.shape[0] * samples.obs.shape[1] * samples.obs.shape[2]
        samples = jax.tree_util.tree_map(lambda x: x.reshape((-1, *x.shape[3:])), samples)
        
        # Shuffle samples
        rng_key, subkey = jax.random.split(rng_key)
        ixs = jax.random.permutation(subkey, jnp.arange(samples.obs.shape[0]))
        samples = jax.tree_util.tree_map(lambda x: x[ixs], samples)
        
        # Create minibatches
        num_updates = samples.obs.shape[0] // config.training_batch_size
        batch_per_device = config.training_batch_size // num_devices
        minibatches = jax.tree_util.tree_map(
            lambda x: x[:num_updates * config.training_batch_size].reshape((num_updates, num_devices, batch_per_device) + x.shape[1:]), 
            samples
        )

        # Training
        print(f"Training on {num_updates} minibatches...")
        policy_losses, value_losses = [], []
        for i in range(num_updates):
            minibatch = jax.tree_util.tree_map(lambda x: x[i], minibatches)
            model, opt_state, policy_loss, value_loss = train(model, opt_state, minibatch)
            policy_losses.append(policy_loss.mean().item())
            value_losses.append(value_loss.mean().item())
        
        policy_loss = sum(policy_losses) / len(policy_losses)
        value_loss = sum(value_losses) / len(value_losses)

        # Log training metrics
        et = time.time()
        hours += (et - st) / 3600
        log.update(
            {
                "train/policy_loss": policy_loss,
                "train/value_loss": value_loss,
                "hours": hours,
                "frames": frames,
            }
        )
        
        print(f"Iteration {iteration} completed in {et - st:.2f} seconds")

    print("Training complete!") 