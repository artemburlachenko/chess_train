/// <reference types="vite/client" />

// Define vue file types
declare module '*.vue' {
  import type { DefineComponent } from 'vue'
  // eslint-disable-next-line @typescript-eslint/no-explicit-any, @typescript-eslint/ban-types
  const component: DefineComponent<{}, {}, any>
  export default component
}

// Define svg file type
declare module '*.svg' {
  const content: string
  export default content
} 