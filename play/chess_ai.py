#!/usr/bin/env python3
"""Chess AI module for making single moves"""

import os
import pickle
import jax
import jax.numpy as jnp
import pgx
import haiku as hk
import sys
from typing import Tuple

# Add imports for pgx chess utilities
try:
    from pgx.experimental import chess as pgx_chess
    HAS_EXPERIMENTAL_CHESS = True
    print("Using pgx.experimental.chess for FEN support")
except ImportError:
    HAS_EXPERIMENTAL_CHESS = False
    print("pgx.experimental.chess not available - upgrade to pgx >= 2.1.0 for FEN support")

# Cache for loaded models to avoid reloading the same checkpoint
_MODEL_CACHE = {}

# Set JAX to use BF16 precision by default
jax.config.update("jax_default_matmul_precision", 'bfloat16')
# use max 65% of memory
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".35"

# 2. Не предвыделять этот объём целиком при старте,
#    а брать память постепенно по мере необходимости
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# Define custom dtype for BF16
BF16 = jnp.bfloat16
# Add a placeholder for Config for pickle loading
class Config:
    pass

# Register Config class with pickle
# This maps the original Config class from chess_train to our placeholder
try:
    from chess_train import Config as OriginalConfig
    sys.modules['__main__'].Config = OriginalConfig
except ImportError:
    # For standalone use, create dummy Config
    print("chess_train module not found, using dummy Config")

def convert_params_to_bfloat16(params):
    """Convert all parameters to bfloat16."""
    return jax.tree_util.tree_map(
        lambda x: x.astype(jnp.bfloat16) if hasattr(x, "dtype") and x.dtype != jnp.int32 and x.dtype != jnp.bool_ else x, 
        params
    )

class AZNet(hk.Module):
    """AlphaZero neural network for chess."""
    def __init__(
        self,
        num_actions,
        num_channels: int = 128,
        num_blocks: int = 6,
        resnet_v2: bool = True,
        name="chess_az_net",
    ):
        super().__init__(name=name)
        self.num_actions = num_actions
        self.num_channels = num_channels
        self.num_blocks = num_blocks
        self.resnet_v2 = resnet_v2
        
    def __call__(self, x, is_training, test_local_stats):
        # Convert input to bfloat16
        x = x.astype(jnp.bfloat16)
        
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
        
        # Final conversion to bfloat16
        policy = policy.astype(jnp.bfloat16)
        value = value.astype(jnp.bfloat16)
        
        return policy, value

def load_checkpoint(checkpoint_path):
    """Load a model checkpoint and convert parameters to bfloat16.
    
    Includes caching to avoid reloading the same model multiple times.
    """
    # Check if model is already in cache
    if checkpoint_path in _MODEL_CACHE:
        print(f"Using cached model for {checkpoint_path}")
        return _MODEL_CACHE[checkpoint_path]
    
    print(f"Loading checkpoint: {checkpoint_path}")
    # Check if file exists
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        
    with open(checkpoint_path, 'rb') as f:
        data = pickle.load(f)
    
    # Convert model parameters to bfloat16
    print("Converting parameters to bfloat16...")
    model_params, model_state = data["model"]
    model_params = convert_params_to_bfloat16(model_params)
    model_state = convert_params_to_bfloat16(model_state)
    data["model"] = (model_params, model_state)
    
    # Cache the model
    _MODEL_CACHE[checkpoint_path] = data
    
    return data

def get_forward_fn(env):
    """Get the forward function for the neural network."""
    return hk.without_apply_rng(hk.transform_with_state(lambda x, is_eval=False: AZNet(
        num_actions=env.num_actions,
        num_channels=128,
        num_blocks=6,
        resnet_v2=True,
    )(x, is_training=not is_eval, test_local_stats=False)))

def get_uci_move(fen: str, checkpoint_path: str, num_simulations: int = 0, use_mcts: bool = False) -> str:
    """
    Get the next move from the AI and return the resulting FEN position.
    
    Args:
        fen: FEN string representing the current board position
        checkpoint_path: Path to the model checkpoint to use
        num_simulations: Number of MCTS simulations to run (0 for direct policy)
        use_mcts: Whether to use MCTS search
        
    Returns:
        A string with the FEN after the AI's move or error message
    """
    # Check if pgx.experimental.chess is available
    if not HAS_EXPERIMENTAL_CHESS:
        print("ERROR: pgx.experimental.chess is required for FEN support")
        return "error:pgx_chess_not_available"
    
    # Set up environment
    env = pgx.make("chess")
    
    # Get environment functions
    init_fn = jax.jit(env.init)
    step_fn = jax.jit(env.step)
    
    # Get forward function
    forward_fn = get_forward_fn(env)
    
    # Load checkpoint
    try:
        data = load_checkpoint(checkpoint_path)
        model_params, model_state = jax.device_put(data["model"])
    except Exception as e:
        error_msg = f"ERROR: Failed to load checkpoint: {e}"
        print(error_msg)
        return f"error:checkpoint_load_failed"
    
    # Initialize state from FEN
    try:
        # Use pgx_chess.from_fen to get the state
        state = pgx_chess.from_fen(fen)
        print(f"Successfully loaded position from FEN")
    except Exception as e:
        error_msg = f"ERROR: Failed to load FEN position: {e}"
        print(error_msg)
        return f"error:invalid_fen"
    
    print(f"Current player: {state.current_player} (0=White, 1=Black)")
    
    # Verify that there are legal moves available
    legal_actions_available = jnp.any(state.legal_action_mask)
    if not legal_actions_available:
        error_msg = "ERROR: No legal moves available - game might be over"
        print(error_msg)
        return "error:no_legal_moves"
    
    # Get observation and ensure BF16 precision
    observation = state.observation.astype(BF16)
    
    # Get initial policy and value from model
    try:
        (logits, value), _ = forward_fn.apply(model_params, model_state, observation[None], is_eval=True)
    except Exception as e:
        error_msg = f"ERROR: Model inference failed: {e}"
        print(error_msg)
        return "error:model_inference_failed"
    
    # Use MCTS if enabled and simulations >= 2
    if use_mcts and num_simulations >= 2:
        try:
            import mctx
            # MCTS approach
            # Create batch with single state
            batch_state = jax.tree_util.tree_map(lambda x: jnp.expand_dims(x, axis=0), state)
            
            # Create the recurrent function for MCTS that supports batched operations
            def mcts_recurrent_fn(model, rng_key, action, state):
                """Recurrent function wrapper for MCTS."""
                del rng_key
                model_params, model_state = model

                # Store current player before step
                current_player = state.current_player
                
                # Apply action to environment - using correct batched version
                next_state = jax.vmap(step_fn)(state, action)
                
                # Get policy and value from model
                (logits, value), _ = forward_fn.apply(
                    model_params, model_state, next_state.observation, is_eval=True
                )
                
                # Mask invalid actions
                logits = logits - jnp.max(logits, axis=-1, keepdims=True)
                logits = jnp.where(next_state.legal_action_mask, logits, jnp.finfo(BF16).min)
                
                # Get rewards and discount
                reward = next_state.rewards[jnp.arange(next_state.rewards.shape[0]), current_player].astype(BF16)
                value = jnp.where(next_state.terminated, jnp.zeros_like(value, dtype=BF16), value)
                discount = (-1.0 * jnp.ones_like(value)).astype(BF16)
                discount = jnp.where(next_state.terminated, jnp.zeros_like(discount, dtype=BF16), discount)
                
                # Create MCTS output
                recurrent_fn_output = mctx.RecurrentFnOutput(
                    reward=reward,
                    discount=discount,
                    prior_logits=logits,
                    value=value
                )
                
                return recurrent_fn_output, next_state
            
            # Create properly structured root for MCTS
            root = mctx.RootFnOutput(
                prior_logits=logits,  # Already has batch dimension from forward_fn
                value=value,          # Already has batch dimension from forward_fn
                embedding=batch_state  # Batch dimension added above
            )
            
            # Run MCTS
            print(f"Running MCTS with {num_simulations} simulations...")
            rng = jax.random.PRNGKey(0)
            rng, mcts_key = jax.random.split(rng)
            
            policy_output = mctx.gumbel_muzero_policy(
                params=(model_params, model_state),
                rng_key=mcts_key,
                root=root,
                recurrent_fn=mcts_recurrent_fn,
                num_simulations=num_simulations,
                invalid_actions=~batch_state.legal_action_mask,
                qtransform=mctx.qtransform_completed_by_mix_value,
                gumbel_scale=jnp.array(1.0, dtype=BF16),
            )
            
            # Make sure we only select from legal actions
            # Filter policy by legal actions
            legal_policy = jnp.where(batch_state.legal_action_mask[0], 
                                  policy_output.action_weights[0], 
                                  jnp.finfo(BF16).min)
            
            # Extract action from first batch element - guaranteed to be legal
            action = jnp.argmax(legal_policy)
            
            print(f"MCTS complete, selected action: {int(action)}")
        except Exception as e:
            error_msg = f"ERROR: MCTS failed: {e}"
            print(error_msg)
            return f"error:mcts_failed"
    else:
        # Standard approach (direct policy)
        # Make sure we mask out illegal actions
        masked_logits = jnp.where(state.legal_action_mask, logits[0], jnp.finfo(BF16).min)
        
        # Select action using argmax - guaranteed to be legal
        action = jnp.argmax(masked_logits)
    
    # Double-check that the action is legal
    if not state.legal_action_mask[action]:
        error_msg = f"ERROR: Selected action {action} is not legal!"
        print(error_msg)
        return f"error:illegal_action_{action}"
    
    # Apply the action to get the next state
    next_state = step_fn(state, action)
    
    # Convert the next state to FEN
    try:
        next_fen = pgx_chess.to_fen(next_state)
        print(f"New position after move: {next_fen}")
        
        # Return the FEN string as the result
        return next_fen
    except Exception as e:
        error_msg = f"ERROR: Failed to convert state to FEN: {e}"
        print(error_msg)
        return f"error:fen_conversion_failed"

def action_to_uci(action_id: int, state=None) -> str:
    """
    Stub function that returns an error message.
    This API is deprecated - use get_uci_move to get FEN strings instead.
    
    Args:
        action_id: The action ID from PGX
        state: The current board state (optional)
        
    Returns:
        An error string
    """
    print(f"WARNING: action_to_uci is deprecated - use get_uci_move to get FEN strings instead")
    return f"error:deprecated_api"

def action_to_uci_basic(action_id: int, state=None) -> str:
    """
    Legacy function - now just calls the new action_to_uci function.
    """
    return action_to_uci(action_id, state) 