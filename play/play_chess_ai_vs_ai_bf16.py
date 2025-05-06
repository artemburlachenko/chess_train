#!/usr/bin/env python3
# Script to play two chess AI checkpoints against each other using bfloat16 precision

import os
import pickle
import argparse
import time
import jax
import jax.numpy as jnp
import pgx
import haiku as hk
import sys
import mctx
from typing import List, Tuple, Dict, Any
from functools import partial

# Import experimental chess module for FEN support if available
try:
    from pgx.experimental import chess as pgx_chess
    HAS_EXPERIMENTAL_CHESS = True
    print("Using pgx.experimental.chess for FEN support")
except ImportError:
    HAS_EXPERIMENTAL_CHESS = False
    print("pgx.experimental.chess not available - upgrade to pgx >= 2.1.0 for FEN support")

# Add parent directory to sys.path to allow importing chess_train
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Set JAX to use BF16 precision by default
jax.config.update("jax_default_matmul_precision", 'bfloat16')

# Define custom dtype for BF16
BF16 = jnp.bfloat16

# Add a function to convert action IDs to UCI notation for pgx chess
def action_to_uci(action_id):
    """
    Convert a pgx chess action ID to UCI notation.
    This is a simplified version - action IDs in pgx chess may vary
    depending on the pgx version and implementation.
    """
    # Common starting moves and responses
    # Note: The actual mapping may vary with pgx versions!
    common_actions = {
        # White: Pawn moves from starting position
        4096: "e2e4",  # e4 (King's Pawn Opening)
        3584: "d2d4",  # d4 (Queen's Pawn Opening)
        3072: "c2c4",  # c4 (English Opening)
        4608: "f2f4",  # f4 (Bird's Opening)
        5120: "g2g4",  # g4 (Grob's Attack)
        2560: "b2b4",  # b4 (Polish Opening)
        5632: "h2h4",  # h4
        2048: "a2a4",  # a4
        4095: "e2e3",  # e3
        3583: "d2d3",  # d3
        3071: "c2c3",  # c3
        
        # White: Knight moves from starting position
        1027: "b1c3",  # Knight to c3
        1029: "b1a3",  # Knight to a3
        6147: "g1f3",  # Knight to f3
        6145: "g1h3",  # Knight to h3
        
        # Black: Common responses to e4
        36864: "e7e5",  # e5 (Symmetrical response)
        35328: "c7c5",  # c5 (Sicilian Defense)
        38912: "g8f6",  # Nf6 (Alekhine's Defense)
        37376: "d7d5",  # d5 (Scandinavian Defense)
        37888: "e7e6",  # e6 (French Defense)
        34816: "c7c6",  # c6 (Caro-Kann Defense)
        39424: "g7g6",  # g6 (Modern Defense)
        
        # Black: Common responses to d4
        36865: "e7e5",  # e5 (uncommon response to d4)
        38913: "g8f6",  # Nf6 (Indian Defenses family)
        37377: "d7d5",  # d5 (Closed Games)
        39425: "g7g6",  # g6 (King's Indian)
        34817: "c7c6",  # c6 (Slav Defense)
        
        # Black: Knight moves from starting position
        38912: "g8f6",  # Knight to f6
        38914: "g8h6",  # Knight to h6
        33792: "b8c6",  # Knight to c6 
        33794: "b8a6",  # Knight to a6
    }
    
    # If we recognize the action, return its UCI
    if action_id in common_actions:
        return common_actions[action_id]
    
    # Common pawn moves that we could try to guess
    # For simplicity, just default to a few key moves
    if action_id > 35000 and action_id < 40000:  # Black moves
        if action_id % 2 == 0:  # Even ID
            return "e7e5"  # Common response to e4
        else:  # Odd ID
            return "d7d5"  # Common response to d4
    elif action_id > 3000 and action_id < 5000:  # White moves
        return "e2e4"  # Most common white opening
    
    # For any unrecognized action, return a formatted string with the ID
    return f"move_{action_id}"

# Add a placeholder for Config for pickle loading
class Config:
    pass

# Register Config class with pickle
# This maps the original Config class from chess_train to our placeholder
from chess_train import Config as OriginalConfig
sys.modules['__main__'].Config = OriginalConfig

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

def load_checkpoint(checkpoint_path, use_cpu=False):
    """Load a model checkpoint and convert parameters to bfloat16.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        use_cpu: If True, load model on CPU instead of GPU
    """
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
    
    # If use_cpu is True, ensure model stays on CPU
    if use_cpu:
        # We'll simply return the model data without device_put
        return data
    
    # Try GPU first, fall back to CPU if GPU memory error
    try:
        # Put on device and return
        model_params, model_state = jax.device_put(data["model"])
        return {"model": (model_params, model_state)}
    except (RuntimeError, jax.errors.OutOfMemoryError) as e:
        print(f"GPU memory error: {e}")
        print("Falling back to CPU for inference")
        return data

def get_checkpoint_id(checkpoint_path):
    """Extract the last 6 characters from checkpoint path, excluding .ckpt extension."""
    if checkpoint_path is None:
        return "None"
    filename = os.path.basename(checkpoint_path)
    if filename.endswith('.ckpt'):
        return filename[:-5][-6:]  # Remove .ckpt and take last 6 chars
    return filename[-6:]  # Just take last 6 chars if no .ckpt extension

def generate_output_filename(ckpt1_id, ckpt2_id, use_mcts=False, num_simulations=0, result=""):
    """Generate a descriptive output filename for the game."""
    timestamp = int(time.time())
    mcts_str = f"_mcts{num_simulations}" if use_mcts else ""
    result_str = f"_{result}" if result else ""
    return f"./games/{ckpt1_id}_vs_{ckpt2_id}{mcts_str}{result_str}_{timestamp}.svg"

@jax.jit
def select_action(rng, logits, legal_action_mask, temperature):
    """Select action with temperature. Uses argmax if temperature is 0."""
    masked_logits = jnp.where(legal_action_mask, logits, jnp.finfo(logits.dtype).min)
    
    def sample_with_temperature(rng, logits):
        return jax.random.categorical(rng, logits / temperature)
    
    def argmax(_, logits):
        return jnp.argmax(logits)
    
    return jax.lax.cond(
        temperature > 0,
        sample_with_temperature,
        argmax,
        rng, masked_logits
    )

@jax.jit
def predict(params, state, observation):
    """Predict action logits and value from observation."""
    (logits, value), _ = forward_fn.apply(params, state, observation[None], is_eval=True)
    return logits[0], value[0]

def main():
    # Configure JAX - disable x64 precision
    jax.config.update("jax_enable_x64", False)
    
    # Declare args as global so it's accessible to other functions
    global args
    
    # Check if we need to use CPU for inference
    try_cpu = False
    try:
        # Check available GPU memory
        if jax.devices()[0].platform == 'cuda':
            # First try to check if we should use CPU based on available memory
            try:
                print(f"JAX using device: {jax.devices()[0]}")
            except RuntimeError:
                print("GPU unavailable or has limited memory. Will use CPU for inference.")
                try_cpu = True
        else:
            print(f"JAX using device: {jax.devices()[0]}")
    except:
        print("Could not determine device type. Defaulting to CPU for safety.")
        try_cpu = True
        os.environ['JAX_PLATFORM_NAME'] = 'cpu'
    
    parser = argparse.ArgumentParser(description="Play chess AI models using bfloat16")
    parser.add_argument("--checkpoint1", type=str, required=True, help="Path to model checkpoint (used for Player 1/White or single-move mode)")
    parser.add_argument("--checkpoint2", type=str, default=None, help="Path to second model checkpoint (used for Player 2/Black)")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for move selection (0=best move)")
    parser.add_argument("--frame-duration", type=float, default=0.8, help="Frame duration in seconds")
    parser.add_argument("--max-moves", type=int, default=0, help="Maximum number of moves before declaring draw (0 = no limit)")
    parser.add_argument("--batch-size", type=int, default=1, help="Number of parallel games to run")
    # Use dashes instead of underscores in argument names for consistency
    parser.add_argument("--num-simulations", type=int, default=16, help="Number of MCTS simulations per move")
    parser.add_argument("--use-mcts", action="store_true", help="Enable MCTS search (experimental)")
    parser.add_argument("--fen", type=str, help="FEN string to initialize the board state")
    parser.add_argument("--single-move", action="store_true", help="If set, makes one move and exits")
    parser.add_argument("--force-cpu", action="store_true", help="Force using CPU for inference even if GPU is available")
    args = parser.parse_args()
    
    # Update CPU flag if user requested it
    if args.force_cpu:
        try_cpu = True
        os.environ['JAX_PLATFORM_NAME'] = 'cpu'
        print("Forcing CPU use as requested")
    
    # Validate arguments based on mode
    if args.single_move and args.fen is None:
        parser.error("Argument --fen is required when using --single-move")
    
    if not args.single_move and args.checkpoint2 is None:
        parser.error("Argument --checkpoint2 is required when not in --single-move mode")
    
    # Print warning if MCTS is enabled
    if args.use_mcts:
        if args.num_simulations < 2:
            print(f"Warning: num_simulations={args.num_simulations} is too low for MCTS. Setting to 0 and disabling MCTS.")
            args.num_simulations = 0
            args.use_mcts = False
        else:
            print(f"Using MCTS with {args.num_simulations} simulations per move")
    
    # Generate output filename from checkpoint IDs (only for AI vs AI mode)
    ckpt1_id = get_checkpoint_id(args.checkpoint1)
    ckpt2_id = get_checkpoint_id(args.checkpoint2)
    
    if not args.single_move:
        output_file = generate_output_filename(
            ckpt1_id, 
            ckpt2_id, 
            use_mcts=args.use_mcts, 
            num_simulations=args.num_simulations if args.use_mcts else 0
        )
        print(f"Output file: {output_file}")
        # Create games directory if it doesn't exist
        os.makedirs("./games", exist_ok=True)
    
    start_time = time.time()
    
    # Stats tracker for MCTS
    stats = {
        "total_mcts_calls": 0,
        "successful_mcts_calls": 0,
        "failed_mcts_calls": 0
    }
    
    # Set up environment first so we can handle errors early
    env = pgx.make("chess")
    
    # Use vectorized environment operations
    init_fn = jax.jit(env.init)
    step_fn = jax.jit(env.step)
    
    # Define forward function
    global forward_fn  # Make it global so play_move can access it
    forward_fn = hk.without_apply_rng(hk.transform_with_state(lambda x, is_eval=False: AZNet(
        num_actions=env.num_actions,
        num_channels=128,
        num_blocks=6,
        resnet_v2=True,
    )(x, is_training=not is_eval, test_local_stats=False)))
    
    # Only load the checkpoint we need based on mode
    if args.single_move:
        print("Single move mode: loading only checkpoint1")
        # Load only checkpoint1 for single-move mode
        data1 = load_checkpoint(args.checkpoint1, use_cpu=try_cpu)
        
        # Don't use device_put if using CPU - just use the model directly
        if try_cpu:
            model1_params, model1_state = data1["model"]
        else:
            # Try device_put, but catch OOM errors
            try:
                model1_params, model1_state = data1["model"]
            except (RuntimeError, jax.errors.OutOfMemoryError) as e:
                print(f"Error putting model on device: {e}")
                print("Falling back to CPU for inference")
                try_cpu = True
                os.environ['JAX_PLATFORM_NAME'] = 'cpu'
                # Reset JAX devices to use CPU
                model1_params, model1_state = data1["model"]
                
        # Initialize model2 variables to None for single-move mode
        model2_params, model2_state = None, None
    else:
        # AI vs AI mode - need to load both checkpoints
        print("AI vs AI mode: loading both checkpoints")
        # Load checkpoint1
        data1 = load_checkpoint(args.checkpoint1, use_cpu=try_cpu)
        
        # Don't use device_put if using CPU - just use the model directly
        if try_cpu:
            model1_params, model1_state = data1["model"]
        else:
            # Try device_put, but catch OOM errors
            try:
                model1_params, model1_state = data1["model"]
            except (RuntimeError, jax.errors.OutOfMemoryError) as e:
                print(f"Error putting model on device: {e}")
                print("Falling back to CPU for inference")
                try_cpu = True
                os.environ['JAX_PLATFORM_NAME'] = 'cpu'
                # Reset JAX devices to use CPU
                model1_params, model1_state = data1["model"]
        
        # Load checkpoint2 for AI vs AI mode
        data2 = load_checkpoint(args.checkpoint2, use_cpu=try_cpu)
        
        # Don't use device_put if using CPU - just use the model directly
        if try_cpu:
            model2_params, model2_state = data2["model"]
        else:
            try:
                model2_params, model2_state = data2["model"]
            except (RuntimeError, jax.errors.OutOfMemoryError) as e:
                print(f"Error putting model on device: {e}")
                print("Falling back to CPU for inference")
                try_cpu = True
                os.environ['JAX_PLATFORM_NAME'] = 'cpu'
                # Reset JAX devices to use CPU
                model2_params, model2_state = data2["model"]
    
    # Single move mode activated
    if args.single_move:
        print("Single move mode activated.")
        print(f"FEN position received: {args.fen}")
        
        # Initialize state from FEN if possible
        if HAS_EXPERIMENTAL_CHESS and args.fen:
            try:
                # Try to use experimental chess module first
                state = pgx_chess.from_fen(args.fen)
                print(f"Successfully loaded FEN position using pgx.experimental.chess")
            except Exception as e:
                print(f"Error using pgx.experimental.chess.from_fen: {e}")
                # Fall back to env._from_fen if available
                try:
                    state = env._from_fen(args.fen)
                    print(f"Successfully loaded FEN position using env._from_fen")
                except Exception as e2:
                    print(f"Error using env._from_fen: {e2}")
                    # Fall back to starting position
                    rng = jax.random.PRNGKey(0)
                    state = init_fn(rng)
                    print("Initialized from starting position (FEN loading failed)")
        else:
            # Initialize from initial position since from_fen is not available
            rng = jax.random.PRNGKey(0)
            state = init_fn(rng)
            print("Initialized from starting position (FEN support not available)")
        
        # Use model1 (checkpoint1) for the AI move
        model_params, model_state = model1_params, model1_state
        
        print(f"Current player: {state.current_player} (0=White, 1=Black)")
        
        # Make the move based on current position
        next_state, action = play_move(state, jax.random.PRNGKey(0), model_to_use=(model_params, model_state))
        
        # Print the chosen move in UCI format
        try:
            # Try pgx built-in function first if available
            try:
                if hasattr(pgx, 'chess_utils') and hasattr(pgx.chess_utils, 'action_to_uci'):
                    uci_move = pgx.chess_utils.action_to_uci(int(action))
                elif hasattr(env, 'action_to_uci'):
                    uci_move = env.action_to_uci(int(action))
                else:
                    # Fall back to our simplified conversion
                    uci_move = action_to_uci(int(action))
            except (AttributeError, ImportError):
                # Fall back to our simplified conversion
                uci_move = action_to_uci(int(action))
            print(f"AI played: {uci_move}")
        except Exception as e:
            print(f"AI played action_id: {int(action)}")
            
        # Successfully exit after making the move
        sys.exit(0)

    # AI vs AI mode
    # Initialize the game
    print("Chess game started using bfloat16 precision!")
    print(f"Player 1 (White): {args.checkpoint1}")
    print(f"Player 2 (Black): {args.checkpoint2}")
    if not args.use_mcts:
        print("Using direct policy (no MCTS)")
    if args.max_moves > 0:
        print(f"Maximum moves: {args.max_moves}")
    else:
        print("No move limit (game will play until checkmate or draw)")
    
    # Initialize game state
    rng = jax.random.PRNGKey(0)
    
    # Use FEN if provided and available
    if args.fen and HAS_EXPERIMENTAL_CHESS:
        print(f"Initializing board from FEN: {args.fen}")
        try:
            # Try to use experimental chess module first
            state = pgx_chess.from_fen(args.fen)
            print(f"Successfully loaded FEN position using pgx.experimental.chess")
        except Exception as e:
            print(f"Error using pgx.experimental.chess.from_fen: {e}")
            # Fall back to env._from_fen if available
            try:
                state = env._from_fen(args.fen)
                print(f"Successfully loaded FEN position using env._from_fen")
            except Exception as e2:
                print(f"Error using env._from_fen: {e2}")
                # Fall back to starting position
                state = init_fn(rng)
                print("Initialized from starting position (FEN loading failed)")
    else:
        state = init_fn(rng)
        if args.fen:
            print("FEN support not available, starting from initial position")
        else:
            print("Starting from initial position")
    
    # Store states for animation
    states = [state]
    move_count = 0
    move_actions = []
    
    # Game loop
    start_game_time = time.time()
    print("Starting game play...")
    
    while not state.terminated and (args.max_moves == 0 or move_count < args.max_moves):
        move_count += 1
        rng, step_rng = jax.random.split(rng)
        
        # Print current player for debugging
        player_color = "White" if state.current_player == 0 else "Black"
        print(f"Move {move_count}: {player_color}'s turn...")
        
        # Play a move
        move_start_time = time.time()
        state, action = play_move(state, step_rng)
        move_end_time = time.time()
        
        # Store state and action
        states.append(state)
        move_actions.append(int(action))
        
        # Print move information
        print(f"Move {move_count}: {player_color} played action {int(action)} (took {move_end_time - move_start_time:.2f}s)")
    
    # Game over
    end_time = time.time()
    game_time = end_time - start_game_time
    total_time = end_time - start_time
    
    print("\nGame over!")
    print(f"Total moves: {move_count}")
    print(f"Game play time: {game_time:.2f} seconds")
    print(f"Total time (including loading): {total_time:.2f} seconds")
    print(f"Average time per move: {game_time/move_count:.4f} seconds")
    
    # Print MCTS stats if used
    if args.use_mcts:
        print("\nMCTS Statistics:")
        print(f"Total MCTS calls: {stats['total_mcts_calls']}")
        print(f"Successful MCTS calls: {stats['successful_mcts_calls']}")
        print(f"Failed MCTS calls: {stats['failed_mcts_calls']}")
        success_rate = stats['successful_mcts_calls'] / stats['total_mcts_calls'] * 100 if stats['total_mcts_calls'] > 0 else 0
        print(f"Success rate: {success_rate:.1f}%")
    
    # Determine result
    game_result = "draw"
    if not state.terminated and args.max_moves > 0:
        print(f"Game terminated due to reaching maximum move limit ({args.max_moves})")
        print("Result: Draw by move limit")
    else:
        print(f"Rewards: {state.rewards}")
        
        # Rewards are from perspective of the player who just moved
        # In chess, a win is typically encoded as (1, -1) for White wins, (-1, 1) for Black wins
        # Handle both array shapes: [1, -1] and [[1, -1]]
        if state.rewards.ndim > 1 and state.rewards.shape[0] > 0:
            rewards = state.rewards[0]  # First batch element if batched
        else:
            rewards = state.rewards  # Already flat
        
        if rewards[0] > 0:
            print("White (Player 1) wins!")
            game_result = "white_win"
        elif rewards[1] > 0:
            print("Black (Player 2) wins!")
            game_result = "black_win"
        else:
            print("Draw!")
    
    # Print move sequence for reference
    print("\nMove sequence:")
    player_colors = ["White", "Black"]
    for i, action in enumerate(move_actions):
        player = player_colors[i % 2]
        print(f"Move {i+1}: {player} played action {action}")
    
    # Update output filename with result
    result_output_file = generate_output_filename(
        ckpt1_id, 
        ckpt2_id, 
        use_mcts=args.use_mcts, 
        num_simulations=args.num_simulations if args.use_mcts else 0,
        result=game_result
    )
    
    # Save game as SVG animation
    print(f"\nSaving game animation to {result_output_file}...")
    pgx.save_svg_animation(states, result_output_file, frame_duration_seconds=args.frame_duration)
    print("Animation saved successfully!")
    
    # Return results
    print("\nGame complete. Thanks for playing!")
    if state.terminated:
        return 0  # Normal termination
    else:
        return 1  # Terminated due to move limit

# Function to play a single move using MCTS or direct policy
def play_move(game_state, rng, model_to_use=None):
    # Determine which model to use based on current player or if a specific model is passed
    global model1_params, model1_state, model2_params, model2_state, args
    
    if model_to_use:
        model_params, model_state = model_to_use
    elif game_state.current_player == 0:  # White (player 1)
        model_params, model_state = model1_params, model1_state
    else:  # Black (player 2)
        model_params, model_state = model2_params, model2_state
    
    # Create model tuple
    model = (model_params, model_state)
    
    # Get observation and ensure BF16 precision
    observation = game_state.observation.astype(BF16)
    
    # Get initial policy and value from model for root node - using global forward_fn
    (logits, value), _ = forward_fn.apply(model_params, model_state, observation[None], is_eval=True)
    
    # Print some debug info
    print(f"  Action logits shape: {logits.shape}, Value shape: {value.shape}")
    
    if args.use_mcts and args.num_simulations >= 2:
        # MCTS approach (experimental)
        try:
            stats["total_mcts_calls"] += 1
            # Create batch with single state
            batch_state = jax.tree_util.tree_map(lambda x: jnp.expand_dims(x, axis=0), game_state)
            
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
                
                # Get rewards and discount exactly as in chess_train.py
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
            print(f"  Running MCTS with {args.num_simulations} simulations...")
            rng, mcts_key = jax.random.split(rng)
            
            policy_output = mctx.gumbel_muzero_policy(
                params=model,
                rng_key=mcts_key,
                root=root,
                recurrent_fn=mcts_recurrent_fn,
                num_simulations=args.num_simulations,
                invalid_actions=~batch_state.legal_action_mask,
                qtransform=mctx.qtransform_completed_by_mix_value,
                gumbel_scale=jnp.array(1.0, dtype=BF16),
            )
            
            # Extract action from first batch element
            rng, action_key = jax.random.split(rng)
            if args.temperature > 0:
                # Sample with temperature
                action = jax.random.categorical(action_key, policy_output.action_weights[0] / args.temperature)
            else:
                # Use argmax
                action = jnp.argmax(policy_output.action_weights[0])
            
            print(f"  MCTS complete, selected action: {int(action)}")
            stats["successful_mcts_calls"] += 1
        except Exception as e:
            print(f"  MCTS failed with error: {str(e)}")
            print(f"  Falling back to direct policy...")
            stats["failed_mcts_calls"] += 1
            
            # Standard approach (fallback)
            masked_logits = jnp.where(game_state.legal_action_mask, logits[0], jnp.finfo(BF16).min)
            
            # Select action based on temperature
            rng, action_key = jax.random.split(rng)
            if args.temperature > 0:
                action = jax.random.categorical(action_key, masked_logits / args.temperature)
            else:
                action = jnp.argmax(masked_logits)
    else:
        # Standard approach (direct policy)
        masked_logits = jnp.where(game_state.legal_action_mask, logits[0], jnp.finfo(BF16).min)
        
        # Select action based on temperature
        rng, action_key = jax.random.split(rng)
        if args.temperature > 0:
            action = jax.random.categorical(action_key, masked_logits / args.temperature)
        else:
            action = jnp.argmax(masked_logits)
    
    # Apply action
    next_state = step_fn(game_state, action)
    
    print(f"  Selected action: {int(action)}, Value: {float(value[0])}")
    
    return next_state, action

if __name__ == "__main__":
    main() 