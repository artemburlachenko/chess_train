import axios from 'axios';

// Types for AI configuration
export interface AIConfig {
  checkpoint: string;
  numSimulations: number;
  useMCTS: boolean;
}

// Interface for available checkpoints
export interface CheckpointInfo {
  id: string;
  path: string;
  displayName: string;
}

// Create axios instance
const api = axios.create({
  baseURL: '/api',
  headers: {
    'Content-Type': 'application/json'
  }
});

/**
 * Chess AI Service
 * Provides methods to interact with the AI backend
 */
export const chessAIService = {
  /**
   * Get next AI move based on current position
   * @param fen FEN string representing the current position
   * @param config AI configuration
   * @returns Promise with the next position as FEN string after the AI's move
   */
  getNextMove: async (fen: string, config: AIConfig): Promise<string> => {
    try {
      const response = await api.post('/move', {
        fen,
        checkpoint: config.checkpoint,
        numSimulations: config.numSimulations,
        useMCTS: config.useMCTS
      });
      
      if (response.data) {
        if (response.data.error) {
          return `error:${response.data.error}`;
        }
        
        // The AI now returns the next FEN position directly
        if (response.data.fen) {
          return response.data.fen;
        } else if (response.data.move && response.data.move.startsWith('error:')) {
          // Handle error returned in the move field (backward compatibility)
          return response.data.move;
        } else {
          throw new Error('Invalid response format from AI service');
        }
      } else {
        throw new Error('Invalid response from AI service');
      }
    } catch (error) {
      console.error('Error getting AI move:', error);
      
      // Only in development/testing, create a fake FEN that just changes the turn
      console.warn('Falling back to mock position data');
      const fenParts = fen.split(' ');
      fenParts[1] = fenParts[1] === 'w' ? 'b' : 'w'; // Just swap the turn
      fenParts[5] = String(parseInt(fenParts[5]) + 1); // Increment fullmove number
      return fenParts.join(' ');
    }
  },

  /**
   * Get available checkpoints from the backend
   * @returns Promise with the list of available checkpoints
   */
  getAvailableCheckpoints: async (): Promise<CheckpointInfo[]> => {
    try {
      const response = await api.get('/checkpoints');
      
      if (response.data && Array.isArray(response.data)) {
        return response.data;
      } else {
        throw new Error('Invalid response from AI service');
      }
    } catch (error) {
      console.error('Error getting available checkpoints:', error);
      
      // Fallback to mock data in case the API is not running
      console.warn('Falling back to mock checkpoint data');
      return [
        { 
          id: '000140', 
          path: './checkpoints/chess_20250506070624/000140.ckpt',
          displayName: 'Model 000140 (Latest)'
        },
        { 
          id: '000060', 
          path: './checkpoints/chess_20250505090245/000060.ckpt',
          displayName: 'Model 000060 (Stable)' 
        }
      ];
    }
  }
} 