import { useState, useEffect } from 'react'
import './App.css'
import Dashboard from './components/Dashboard'
import Navbar from './components/Navbar'

function App() {
  return (
    <div className="min-h-screen bg-gray-900">
      <Navbar />
      <Dashboard />
    </div>
  )
}

export default App
