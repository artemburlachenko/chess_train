<template>
  <div class="chess-game">
    <div class="chess-container">
      <div class="chess-board-wrapper">
        <TheChessboard
          :board-config="boardConfig"
          @board-created="onBoardCreated"
          @move="onMove"
          @check="onCheck"
          @checkmate="onCheckmate"
          @stalemate="onStalemate"
          @draw="onDraw"
        />
      </div>
      
      <div class="side-panel">
        <div class="card">
          <h2>Game Controls</h2>
          <div class="chess-controls">
            <button @click="resetGame">New Game</button>
            <button @click="undoMove" :disabled="!canUndo">Undo</button>
            <button @click="boardAPI?.toggleOrientation()">Flip Board</button>
            <button @click="forceAIMove" :disabled="aiThinking">Force AI Move</button>
          </div>

          <div v-if="gameStatus" :class="['status-message', statusClass]">
            {{ gameStatus }}
          </div>
        </div>

        <div class="card">
          <h2>AI Settings</h2>
          
          <div class="form-group">
            <label for="checkpoint">AI Model</label>
            <select 
              id="checkpoint" 
              v-model="aiConfig.checkpoint" 
              class="checkpoints-dropdown"
            >
              <option 
                v-for="checkpoint in availableCheckpoints" 
                :key="checkpoint.id" 
                :value="checkpoint.path"
              >
                {{ checkpoint.displayName }}
              </option>
            </select>
          </div>
          
          <div class="form-group">
            <div class="checkbox-container">
              <input 
                type="checkbox" 
                id="use-mcts" 
                v-model="aiConfig.useMCTS"
              >
              <label for="use-mcts">Use MCTS (Monte Carlo Tree Search)</label>
            </div>
          </div>
          
          <div class="slider-container" v-if="aiConfig.useMCTS">
            <label for="simulation-depth">
              Search Depth: {{ aiConfig.numSimulations }} simulations
            </label>
            <input 
              type="range" 
              id="simulation-depth" 
              v-model.number="aiConfig.numSimulations" 
              min="2"
              max="10000" 
              step="1"
            >
          </div>
        </div>

        <div class="card">
          <h2>Game Info</h2>
          <div v-if="currentOpening" class="opening-info">
            <p>Opening: {{ currentOpening }}</p>
          </div>
          <div class="move-history">
            <p>Moves: {{ moveHistory.length }}</p>
            <div class="moves-list">
              <div v-for="(move, index) in moveHistory" :key="index" class="move-item">
                {{ index + 1 }}. {{ move }}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, computed, watch } from 'vue';
import { TheChessboard } from 'vue3-chessboard';
import type { BoardApi, MoveEvent, PieceColor } from 'vue3-chessboard';
import { chessAIService, type AIConfig, type CheckpointInfo } from '../services/chessAI';

// Board configuration
const boardConfig = {
  coordinates: true,
  animation: {
    enabled: true,
    duration: 200,
  },
  movable: {
    color: 'white' as 'white',
    events: {
      after: onMoveAfter
    }
  },
  drawable: {
    enabled: true,
    defaultSnapToValidMove: true,
  },
};

// State variables
const boardAPI = ref<BoardApi | null>(null);
const aiThinking = ref(false);
const gameStatus = ref('');
const statusClass = ref('');
const currentOpening = ref('');
const moveHistory = ref<string[]>([]);
const canUndo = ref(false);

// AI configuration
const availableCheckpoints = ref<CheckpointInfo[]>([]);
const aiConfig = ref<AIConfig>({
  checkpoint: '',
  numSimulations: 0,
  useMCTS: false,
});

// Watch for changes in numSimulations to update useMCTS
watch(() => aiConfig.value.numSimulations, (newSims) => {
  if (newSims === 1) {
    // If user somehow sets it to 1 (e.g. typing), bump to 2 or 0 based on preference.
    // For now, let's set to 2 if they try for 1, assuming they want MCTS.
    // Or, we can prevent 1 by adjusting slider steps/logic.
    // Let's adjust it to 2 if they try to select 1.
    aiConfig.value.numSimulations = 2;
  }
  aiConfig.value.useMCTS = newSims > 0;
});

// Watch for changes in useMCTS to update numSimulations
watch(() => aiConfig.value.useMCTS, (newUseMCTS) => {
  if (newUseMCTS && aiConfig.value.numSimulations === 0) {
    // If MCTS is turned on and sims are 0, set to a default MCTS value (e.g., 32 or min 2)
    aiConfig.value.numSimulations = 32; // Or 2, as per new minimum for MCTS ON
  } else if (!newUseMCTS) {
    // If MCTS is turned off, set sims to 0
    aiConfig.value.numSimulations = 0;
  }
});

// Board event handlers
function onBoardCreated(api: BoardApi) {
  boardAPI.value = api;
  canUndo.value = false;
  currentOpening.value = '';
}

function onMoveAfter() {
  // Update game info after a move
  if (boardAPI.value) {
    currentOpening.value = '';
    moveHistory.value = boardAPI.value.getHistory();
    canUndo.value = moveHistory.value.length > 0;
  }
}

function onMove(move: MoveEvent) {
  console.log('Move:', move);
  
  // If it's black's turn after move (AI's turn), make AI move
  if (boardAPI.value && boardAPI.value.getFen().split(' ')[1] === 'b') {
    makeAIMove();
  }
}

function onCheck(color: PieceColor) {
  gameStatus.value = `${color === 'white' ? 'White' : 'Black'} is in check!`;
  statusClass.value = 'thinking';
}

function onCheckmate(color: PieceColor) {
  gameStatus.value = `${color === 'white' ? 'White' : 'Black'} wins by checkmate!`;
  statusClass.value = 'success';
}

function onStalemate() {
  gameStatus.value = 'Game drawn by stalemate!';
  statusClass.value = 'thinking';
}

function onDraw() {
  gameStatus.value = 'Game drawn!';
  statusClass.value = 'thinking';
}

// Game functions
async function makeAIMove() {
  if (!boardAPI.value || aiThinking.value) return;
  
  try {
    aiThinking.value = true;
    gameStatus.value = 'AI is thinking...';
    statusClass.value = 'thinking';

    // Get the current position as FEN
    const fen = boardAPI.value.getFen();
    console.log(`Making AI move with FEN: ${fen}`);
    
    // Check if it's actually black's turn (AI's turn)
    const currentTurn = fen.split(' ')[1];
    if (currentTurn !== 'b') {
      console.log("Not AI's turn (black) - skipping AI move");
      aiThinking.value = false;
      return;
    }
    
    // Get AI move from the service - this will now return a FEN string
    const nextFen = await chessAIService.getNextMove(fen, aiConfig.value);
    console.log(`AI returned new position: ${nextFen}`);
    
    // Check if we got an error instead of a FEN
    if (nextFen.startsWith('error:')) {
      console.error(`AI error: ${nextFen}`);
      throw new Error(`AI error: ${nextFen}`);
    }
    
    // Set the new position on the board directly
    if (boardAPI.value) {
      try {
        // Use setPosition to update the board directly with the new FEN
        boardAPI.value.setPosition(nextFen);
        
        // Update status
        gameStatus.value = 'AI made a move';
        statusClass.value = 'success';
        
        // Update game state
        moveHistory.value = boardAPI.value.getHistory();
        canUndo.value = moveHistory.value.length > 0;
        
        return;
      } catch (error) {
        console.error('Error updating board position:', error);
        gameStatus.value = 'Error applying AI move';
        statusClass.value = 'error';
      }
    }
  } catch (error) {
    console.error('AI move error:', error);
    gameStatus.value = 'Error making AI move';
    statusClass.value = 'error';
  } finally {
    aiThinking.value = false;
  }
}

// New function to explicitly trigger AI move
async function forceAIMove() {
  if (aiThinking.value) {
    console.log("AI is already thinking, cannot force another move.");
    return;
  }
  if (!boardAPI.value) {
    console.error("Board API not available to force AI move.");
    return;
  }
  
  // Check if it's black's turn
  const fen = boardAPI.value.getFen();
  const currentTurn = fen.split(' ')[1];
  
  if (currentTurn === 'b') {
    console.log("Forcing AI (black) to make a move.");
    await makeAIMove();
  } else {
    console.log("It's white's turn, not forcing AI move.");
    gameStatus.value = "It's your turn to move";
    statusClass.value = 'thinking';
  }
}

function resetGame() {
  if (boardAPI.value) {
    boardAPI.value.resetBoard();
    moveHistory.value = [];
    canUndo.value = false;
    gameStatus.value = 'New game started';
    statusClass.value = 'success';
    currentOpening.value = '';
  }
}

function undoMove() {
  if (boardAPI.value && moveHistory.value.length > 0) {
    // Undo twice to handle both AI and player move
    boardAPI.value.undoLastMove();
    boardAPI.value.undoLastMove();
    moveHistory.value = boardAPI.value.getHistory();
    canUndo.value = moveHistory.value.length > 0;
    currentOpening.value = '';
    gameStatus.value = 'Move undone';
    statusClass.value = 'thinking';
  }
}

// Initialize
onMounted(async () => {
  try {
    // Load available checkpoints
    console.log('Loading checkpoints...');
    availableCheckpoints.value = await chessAIService.getAvailableCheckpoints();
    console.log('Loaded checkpoints:', availableCheckpoints.value);
    
    if (availableCheckpoints.value.length > 0) {
      // Set the first checkpoint as default
      aiConfig.value.checkpoint = availableCheckpoints.value[0].path;
      console.log('Selected checkpoint:', aiConfig.value.checkpoint);
    } else {
      console.error('No checkpoints found!');
      gameStatus.value = 'Error: No AI models found';
      statusClass.value = 'error';
    }
    
    gameStatus.value = 'Game ready. You play as White.';
    statusClass.value = 'success';
  } catch (error) {
    console.error('Error initializing game:', error);
    gameStatus.value = 'Error initializing game';
    statusClass.value = 'error';
  }
});
</script>

<style scoped>
.chess-game {
  width: 100%;
}

.chess-container {
  display: flex;
  gap: 2rem;
  align-items: flex-start;
}

.chess-board-wrapper {
  flex: 0 0 auto;
  min-width: 480px;
}

.side-panel {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.form-group {
  margin-bottom: 1rem;
}

.form-group label {
  display: block;
  margin-bottom: 0.5rem;
  font-weight: 500;
}

.checkbox-container {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.move-history {
  max-height: 200px;
  overflow-y: auto;
}

.moves-list {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
}

.move-item {
  background-color: var(--background-color);
  padding: 0.25rem 0.5rem;
  border-radius: 4px;
  font-family: monospace;
}

/* Responsive adjustments */
@media (max-width: 992px) {
  .chess-container {
    flex-direction: column;
  }
  
  .chess-board-wrapper {
    margin: 0 auto;
  }
}
</style> 