<template>
  <div class="app">
    <header>
      <div class="container">
        <h1>Chess Train - Play Against AI</h1>
      </div>
    </header>

    <main class="container">
      <ChessGame />
    </main>

    <footer>
      <div class="container">
        <p>&copy; {{ new Date().getFullYear() }} Chess Train</p>
      </div>
    </footer>
  </div>
</template>

<script setup lang="ts">
import ChessGame from './components/ChessGame.vue';
</script>

<style>
.app {
  display: flex;
  flex-direction: column;
  min-height: 100vh;
}

header {
  background-color: var(--primary-color);
  color: white;
  padding: 1rem 0;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

main {
  flex: 1;
  padding: 2rem 0;
}

footer {
  background-color: var(--secondary-color);
  color: white;
  padding: 1rem 0;
  text-align: center;
  margin-top: auto;
}
</style> 