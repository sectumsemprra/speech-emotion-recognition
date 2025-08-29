const express = require('express');
const multer = require('multer');
const cors = require('cors');
const path = require('path');
const fs = require('fs');
const { spawn } = require('child_process');

const app = express();
const PORT = 3001;

// Middleware
app.use(cors());
app.use(express.json());

// Create uploads directory if it doesn't exist
const uploadsDir = path.join(__dirname, 'uploads');
if (!fs.existsSync(uploadsDir)) {
  fs.mkdirSync(uploadsDir);
}

// Configure multer for file uploads
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, uploadsDir);
  },
  filename: (req, file, cb) => {
    const timestamp = Date.now();
    cb(null, `audio_${timestamp}.webm`);
  }
});

const upload = multer({ 
  storage: storage,
  limits: {
    fileSize: 50 * 1024 * 1024 // 50MB limit
  }
});

// Function to convert WebM to WAV using FFmpeg
function convertToWav(inputPath, outputPath) {
  return new Promise((resolve, reject) => {
    const ffmpeg = spawn('ffmpeg', [
      '-i', inputPath,
      '-acodec', 'pcm_s16le',
      '-ar', '22050', // Sample rate that works well with most models
      '-ac', '1', // Mono channel
      '-y', // Overwrite output file
      outputPath
    ]);

    ffmpeg.on('close', (code) => {
      if (code === 0) {
        resolve(outputPath);
      } else {
        reject(new Error(`FFmpeg process exited with code ${code}`));
      }
    });

    ffmpeg.on('error', (error) => {
      reject(error);
    });
  });
}

// Function to run emotion detection using Python model
function runEmotionDetection(wavPath) {
  return new Promise((resolve, reject) => {
    // Adjust the path to your dsp.ipynb converted to a Python script
    const pythonScript = path.join(__dirname, 'emotion_detection.py');
    
    const python = spawn('python', [pythonScript, wavPath]);
    
    let result = '';
    let error = '';
    
    python.stdout.on('data', (data) => {
      result += data.toString();
    });
    
    python.stderr.on('data', (data) => {
      error += data.toString();
    });
    
    python.on('close', (code) => {
      if (code === 0) {
        try {
          const emotionData = JSON.parse(result.trim());
          resolve(emotionData);
        } catch (parseError) {
          reject(new Error(`Failed to parse emotion detection result: ${parseError.message}`));
        }
      } else {
        reject(new Error(`Python script failed: ${error}`));
      }
    });
  });
}

// Routes
app.get('/health', (req, res) => {
  res.json({ 
    status: 'healthy', 
    message: 'Emotion detection API is running',
    ffmpegAvailable: checkFFmpegAvailability(),
    pythonAvailable: checkPythonAvailability()
  });
});

// Check if FFmpeg is available
function checkFFmpegAvailability() {
  try {
    const ffmpeg = spawn('ffmpeg', ['-version']);
    return true;
  } catch (error) {
    return false;
  }
}

// Check if Python is available
function checkPythonAvailability() {
  try {
    const python = spawn('python', ['--version']);
    return true;
  } catch (error) {
    return false;
  }
}

app.post('/predict', upload.single('audio'), async (req, res) => {
  let webmPath = null;
  let wavPath = null;

  try {
    if (!req.file) {
      return res.status(400).json({ 
        error: 'No audio file provided' 
      });
    }

    console.log(`Processing audio file: ${req.file.originalname} (${req.file.size} bytes)`);
    
    webmPath = req.file.path;
    const timestamp = Date.now();
    wavPath = path.join(uploadsDir, `audio_${timestamp}.wav`);

    // Convert WebM to WAV
    console.log('Converting audio to WAV format...');
    await convertToWav(webmPath, wavPath);
    console.log('Audio conversion completed');

    // Run emotion detection
    console.log('Running emotion detection...');
    const emotionResult = await runEmotionDetection(wavPath);
    console.log(`Emotion detected: ${emotionResult.emotion} (${(emotionResult.confidence * 100).toFixed(1)}%)`);

    res.json(emotionResult);

  } catch (error) {
    console.error('Error processing audio:', error);
    
    // Return mock data as fallback
    const mockResult = {
      emotion: "neutral",
      confidence: 0.75,
      topEmotions: [
        { emotion: "neutral", score: 0.75 },
        { emotion: "happy", score: 0.15 },
        { emotion: "sad", score: 0.10 }
      ]
    };

    res.json({
      ...mockResult,
      warning: "Used fallback mock data. Check server logs for details.",
      error: error.message
    });

  } finally {
    // Clean up temporary files
    if (webmPath && fs.existsSync(webmPath)) {
      try {
        fs.unlinkSync(webmPath);
        console.log('Cleaned up WebM file');
      } catch (cleanupError) {
        console.error('Error cleaning up WebM file:', cleanupError);
      }
    }
    
    if (wavPath && fs.existsSync(wavPath)) {
      try {
        fs.unlinkSync(wavPath);
        console.log('Cleaned up WAV file');
      } catch (cleanupError) {
        console.error('Error cleaning up WAV file:', cleanupError);
      }
    }
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
  
  // Check dependencies
  console.log('\n🔍 Checking dependencies:');
  console.log(`FFmpeg available: ${checkFFmpegAvailability()}`);
  console.log(`Python available: ${checkPythonAvailability()}`);
  
  if (!checkFFmpegAvailability()) {
    console.log('⚠️  Warning: FFmpeg not found. Please install FFmpeg to convert audio files.');
  }
  
  if (!checkPythonAvailability()) {
    console.log('⚠️  Warning: Python not found. Please ensure Python is installed and accessible.');
  }
});