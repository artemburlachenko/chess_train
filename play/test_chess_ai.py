#!/usr/bin/env python3
"""Test script for chess_ai module"""

import chess_ai
import pgx
import jax
import numpy as np

# Create a chess environment and a state
env = pgx.make('chess')
init_fn = jax.jit(env.init)
state = init_fn(jax.random.PRNGKey(0))

# ----- Test the new action_to_uci function with known moves -----
print('\n==== Testing new action_to_uci function ====')

# Common opening moves to test
test_actions = [
    1842,   # e2e4 for white
    1258,   # e7e5 for black
    2766,   # g1f3 for white (Knight to f3)
    2794,   # g8f6 for black (Knight to f6)
    1330,   # d2d4 for white
    1222,   # d7d5 for black
]

# Test with valid state (white's turn)
for action in test_actions:
    move = chess_ai.action_to_uci(action, state)
    print(f"Action {action} → {move}")

# Try to modify state to be black's turn
try:
    # This is a hack, may not work depending on state structure
    black_state = state.replace(current_player=1)
    
    print('\n==== Testing with black state ====')
    for action in test_actions:
        move = chess_ai.action_to_uci(action, black_state)
        print(f"Action {action} → {move}")
except Exception as e:
    print('\nCould not switch to black player:', e)

# ----- Get legal moves and test conversion of those -----
print('\n==== Testing with legal moves ====')
legal_indices = jax.numpy.where(state.legal_action_mask)[0]
print(f"Found {len(legal_indices)} legal moves")
print(f"Sample of legal actions: {legal_indices[:5]}")

for action in legal_indices[:5]:  # Test the first 5 legal moves
    move = chess_ai.action_to_uci(int(action), state)
    print(f"Legal action {int(action)} → {move}")

# ----- Simulate an actual AI move -----
print('\n==== Testing get_uci_move function ====')
# Use simple FEN of starting position
start_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

# Find the first checkpoint file in the full directory
import glob
checkpoint_files = glob.glob("full/*.ckpt")
if checkpoint_files:
    checkpoint_path = checkpoint_files[0]
    print(f"Found checkpoint: {checkpoint_path}")
    try:
        uci_move = chess_ai.get_uci_move(start_fen, checkpoint_path)
        print(f"AI recommended move: {uci_move}")
    except Exception as e:
        print(f"Error in get_uci_move: {e}")
else:
    print("No checkpoint files found in 'full' directory. Skipping get_uci_move test.")

# Test with invalid state
print('\nTesting with None state (should show error):')
print(chess_ai.action_to_uci_basic(1258, None))

# Create an empty object without current_player attribute
class EmptyState:
    pass

empty_state = EmptyState()
print('\nTesting with invalid state (no current_player):')
print(chess_ai.action_to_uci_basic(1258, empty_state)) 