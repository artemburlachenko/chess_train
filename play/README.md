# Chess Train UI

A Vue 3 frontend for playing against the Chess Train AI models.

## Features

- Play chess against AI models trained with chess_train.py
- Choose from available AI models (checkpoints)
- Adjust search depth and enable/disable Monte Carlo Tree Search (MCTS)
- View current opening, game status, and move history
- Undo moves and reset the game

## Setup

1. Install dependencies:

```bash
cd play
npm install
```

2. Start the development server:

```bash
npm run dev
```

3. In a separate terminal, start the Flask API server:

```bash
cd play
python api.py
```

## How it Works

The UI is built with Vue 3 and [vue3-chessboard](https://github.com/qwerty084/vue3-chessboard), which provides the interactive chessboard component.

The UI communicates with a Flask API, which interacts with the Python AI to get moves. The API provides:

- `/checkpoints` - Retrieves available AI model checkpoints
- `/move` - Gets the next AI move for a given position

## Requirements

- Node.js 14+
- Python 3.7+
- Flask and Flask-CORS (`pip install flask flask-cors`)
- Trained model checkpoints in the `../checkpoints` directory

## Project Structure

- `src/components/ChessGame.vue` - Main chessboard and game controls
- `src/services/chessAI.ts` - Service for communicating with the Flask API
- `api.py` - Flask API for interfacing with the Python AI

## Production Build

To create a production build:

```bash
npm run build
```

The built files will be in the `dist` directory, which can be served with any static file server. 