const express = require('express');
const multer = require('multer');
const cors = require('cors');
const path = require('path');

const app = express();
const PORT = 3001;

// Middleware
app.use(cors());
app.use(express.json());

// Configure multer for file uploads
const storage = multer.memoryStorage();
const upload = multer({ 
  storage: storage,
  limits: {
    fileSize: 10 * 1024 * 1024 // 10MB limit
  }
});

// Mock emotion detection (replace with real ML model)
function mockEmotionDetection() {
  const emotions = [
    { emotion: 'happy', baseScore: 0.8 },
    { emotion: 'sad', baseScore: 0.7 },
    { emotion: 'angry', baseScore: 0.6 },
    { emotion: 'neutral', baseScore: 0.9 },
    { emotion: 'fear', baseScore: 0.5 },
    { emotion: 'surprise', baseScore: 0.4 },
    { emotion: 'disgust', baseScore: 0.3 }
  ];

  // Simulate random emotion detection
  const selectedEmotion = emotions[Math.floor(Math.random() * emotions.length)];
  
  // Generate realistic scores
  const topEmotions = emotions.map(e => ({
    emotion: e.emotion,
    score: e.emotion === selectedEmotion.emotion 
      ? selectedEmotion.baseScore + Math.random() * 0.2
      : Math.random() * 0.3
  })).sort((a, b) => b.score - a.score);

  // Normalize scores
  const totalScore = topEmotions.reduce((sum, e) => sum + e.score, 0);
  topEmotions.forEach(e => e.score = e.score / totalScore);

  return {
    emotion: topEmotions[0].emotion,
    confidence: topEmotions[0].score,
    topEmotions: topEmotions.slice(0, 5)
  };
}

// Routes
app.get('/health', (req, res) => {
  res.json({ status: 'healthy', message: 'Emotion detection API is running' });
});

app.post('/predict', upload.single('audio'), (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ 
        error: 'No audio file provided' 
      });
    }

    console.log(`Processing audio file: ${req.file.originalname} (${req.file.size} bytes)`);

    // Simulate processing time
    setTimeout(() => {
      const result = mockEmotionDetection();
      console.log(`Detected emotion: ${result.emotion} (${(result.confidence * 100).toFixed(1)}%)`);
      
      res.json(result);
    }, 1000 + Math.random() * 2000); // 1-3 second delay

  } catch (error) {
    console.error('Error processing audio:', error);
    res.status(500).json({ 
      error: 'Failed to process audio file',
      details: error.message 
    });
  }
});

// Error handling
app.use((error, req, res, next) => {
  console.error('Server error:', error);
  res.status(500).json({ 
    error: 'Internal server error',
    details: error.message 
  });
});

app.listen(PORT, () => {
  console.log(`🚀 Emotion Detection API running on http://localhost:${PORT}`);
  console.log(`📊 Health check: http://localhost:${PORT}/health`);
  console.log(`🎭 Predict endpoint: http://localhost:${PORT}/predict`);
});
</action>