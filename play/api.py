#!/usr/bin/env python3
"""API server for chess AI interface"""

import os
import json
import sys
import glob
import re
import time
from typing import List, Dict, Any, Optional

from flask import Flask, request, jsonify
from flask_cors import CORS

# Add parent directory to path to import chess_train
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our chess_ai module for making moves
import chess_ai

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Cache of available checkpoints
_CHECKPOINT_CACHE = None
_CACHE_LAST_UPDATED = 0
_CACHE_TTL = 300  # Cache time-to-live in seconds (5 minutes)

def clear_checkpoint_cache():
    """Clear the checkpoint cache to free memory"""
    global _CHECKPOINT_CACHE, _CACHE_LAST_UPDATED
    _CHECKPOINT_CACHE = None
    _CACHE_LAST_UPDATED = 0
    app.logger.info("Checkpoint cache cleared")

def find_checkpoints() -> List[Dict[str, str]]:
    """Find available checkpoint files in the ./full directory."""
    global _CHECKPOINT_CACHE, _CACHE_LAST_UPDATED
    
    # Check if cache is still valid
    current_time = time.time()
    if _CHECKPOINT_CACHE is not None and (current_time - _CACHE_LAST_UPDATED) < _CACHE_TTL:
        return _CHECKPOINT_CACHE
    
    checkpoints = []
    
    # Check only in ./full directory (relative to api.py)
    full_dir_path = "./full" # Path relative to api.py
    app.logger.info(f"Looking for checkpoints in: {os.path.abspath(full_dir_path)}")

    if os.path.exists(full_dir_path) and os.path.isdir(full_dir_path):
        checkpoint_files = glob.glob(f"{full_dir_path}/*.ckpt")
        app.logger.info(f"Found files: {checkpoint_files}")
        
        for checkpoint_file in checkpoint_files:
            # Extract the ID from the filename (e.g., 000140 from 000140.ckpt or 000025 from 000025_float32.ckpt)
            match = re.search(r'(\d{6})(?:_.*?)?\.ckpt$', os.path.basename(checkpoint_file))
            if match:
                checkpoint_id = match.group(1)
                # Make path relative to the api.py script for consistency
                relative_path = os.path.join("full", os.path.basename(checkpoint_file))
                checkpoints.append({
                    "id": checkpoint_id,
                    "path": relative_path, # Store path relative to api.py
                    "displayName": f"Model {checkpoint_id} (full)"
                })
            else:
                app.logger.warning(f"Could not parse checkpoint ID from: {checkpoint_file}")
    else:
        app.logger.warning(f"Directory not found or not a directory: {os.path.abspath(full_dir_path)}")
            
    # Sort checkpoints by ID in descending order (newest first)
    checkpoints.sort(key=lambda x: x["id"], reverse=True)
    
    _CHECKPOINT_CACHE = checkpoints
    _CACHE_LAST_UPDATED = current_time
    
    if not checkpoints:
        app.logger.warning("No checkpoints found in ./full directory.")
    else:
        app.logger.info(f"Found {len(checkpoints)} checkpoints: {[cp['displayName'] for cp in checkpoints]}")
    return checkpoints

def get_ai_move(fen: str, checkpoint_path: str, num_simulations: int, use_mcts: bool) -> str:
    """Get the next move from the AI using the chess_ai module"""
    try:
        app.logger.info(f"Getting AI move for FEN: {fen}, checkpoint: {checkpoint_path}")
        app.logger.info(f"MCTS: {use_mcts}, simulations: {num_simulations}")
        
        # Get move directly from chess_ai module
        next_fen = chess_ai.get_uci_move(
            fen=fen,
            checkpoint_path=checkpoint_path,
            num_simulations=num_simulations,
            use_mcts=use_mcts
        )
        
        app.logger.info(f"AI returned new position: {next_fen}")
        return next_fen
    except Exception as e:
        app.logger.error(f"Error from chess_ai module: {e}")
        # Return an error message
        app.logger.warning("Returning error message to client")
        return f"error:ai_exception:{str(e)}"

@app.route('/checkpoints', methods=['GET'])
def get_checkpoints():
    """Get a list of available checkpoint files."""
    try:
        force_refresh = request.args.get('refresh', 'false').lower() == 'true'
        if force_refresh:
            clear_checkpoint_cache()
            
        checkpoints = find_checkpoints()
        return jsonify(checkpoints)
    except Exception as e:
        app.logger.error(f"Error getting checkpoints: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/move', methods=['POST'])
def get_move():
    """Get the next move from the AI"""
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        fen = data.get('fen')
        checkpoint = data.get('checkpoint')
        num_simulations = data.get('numSimulations', 0)
        use_mcts = data.get('useMCTS', False)
        
        if not fen:
            return jsonify({"error": "FEN position is required"}), 400
        if not checkpoint:
            return jsonify({"error": "Checkpoint path is required"}), 400
        
        # Validate num_simulations and use_mcts settings
        # If MCTS is enabled but simulations < 2, set simulations to 0 and disable MCTS
        if use_mcts and num_simulations < 2:
            app.logger.warning(f"MCTS enabled but num_simulations={num_simulations} is too low. Setting to 0 and disabling MCTS.")
            num_simulations = 0
            use_mcts = False
        
        # Get next position from AI
        next_fen = get_ai_move(fen, checkpoint, num_simulations, use_mcts)
        
        if next_fen.startswith("error:"):
            # If there was an error, include it in the response
            return jsonify({
                "error": next_fen[6:],  # remove "error:" prefix
                "fen": fen,
                "checkpoint": checkpoint,
                "numSimulations": num_simulations,
                "useMCTS": use_mcts
            })
        
        # Return the new FEN position
        return jsonify({
            "fen": next_fen,
            "originalFen": fen,
            "checkpoint": checkpoint,
            "numSimulations": num_simulations,
            "useMCTS": use_mcts
        })
    except Exception as e:
        app.logger.error(f"Error processing move request: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/clear-cache', methods=['POST'])
def clear_cache():
    """Manually clear the checkpoint cache"""
    try:
        clear_checkpoint_cache()
        return jsonify({"success": True, "message": "Cache cleared successfully"})
    except Exception as e:
        app.logger.error(f"Error clearing cache: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Run the Flask app without preloading checkpoints
    app.run(debug=True, port=5000) 