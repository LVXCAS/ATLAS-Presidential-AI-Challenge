import React from 'react'
import { createRoot } from 'react-dom/client'
import App from './App'

try {
  const root = createRoot(document.getElementById('root')!)
  root.render(<App />)
} catch (error) {
  console.error('Failed to render app:', error)
  document.body.innerHTML = '<div style="color: red; font-family: monospace; padding: 20px;">ERROR: Failed to load trading terminal. Check console for details.</div>'
}
