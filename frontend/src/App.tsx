import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Landing from './pages/Landing';
import DetectEmotion from './pages/DetectEmotion';

function App() {
  return (
    <Router>
      <div className="min-h-screen bg-gradient-to-br from-blue-900 via-purple-900 to-indigo-900">
        <Routes>
          <Route path="/" element={<Landing />} />
          <Route path="/detect" element={<DetectEmotion />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;