# Speech Emotion Recognition App

A full-stack application for real-time speech emotion detection using React frontend and Node.js backend.

## 🎯 Features

- **Audio Recording**: One-click voice recording with real-time feedback
- **Emotion Analysis**: AI-powered emotion detection from voice patterns
- **Modern UI**: Beautiful, responsive interface with smooth animations
- **Real-time Results**: Instant emotion predictions with confidence scores

## 🚀 Quick Start

### Frontend (React + Vite)
```bash
# Navigate to frontend directory
cd frontend

# Install dependencies and start development server
npm install
npm run dev
```

The frontend will run on `http://localhost:5173`

### Backend (Node.js + Express)
```bash
# Navigate to backend directory
cd backend

# Install dependencies
npm install

# Start the API server
npm run dev
```

The backend API will run on `http://localhost:3001`

## 📁 Project Structure

```
├── src/
│   ├── pages/
│   │   ├── Landing.tsx          # Landing page with hero section
│   │   └── DetectEmotion.tsx    # Main emotion detection interface
│   ├── components/
│   │   ├── AudioRecorder.tsx    # Audio recording component
│   │   └── EmotionResult.tsx    # Emotion results display
│   └── App.tsx                  # Main app with routing
├── backend/
│   ├── server.js               # Express API server
│   └── package.json            # Backend dependencies
└── README.md                   # This file
```

## 🔧 API Endpoints

### `POST /predict`
Upload audio file for emotion detection.

**Request:**
- Content-Type: `multipart/form-data`
- Body: Audio file (WAV format recommended)

**Response:**
```json
{
  "emotion": "happy",
  "confidence": 0.85,
  "topEmotions": [
    { "emotion": "happy", "score": 0.85 },
    { "emotion": "neutral", "score": 0.10 },
    { "emotion": "sad", "score": 0.05 }
  ]
}
```

### `GET /health`
Check API status.

## 🔬 Machine Learning Integration

The current backend uses mock emotion detection for demonstration. To integrate real ML model:

1. Replace the `mockEmotionDetection()` function in `backend/server.js`
2. Add your trained model loading and prediction logic
3. Install required ML libraries (e.g., TensorFlow.js, ONNX Runtime)

**Example integration points:**
```javascript
// In server.js, replace mockEmotionDetection with:
async function predictEmotion(audioBuffer) {
  // Load your trained model
  // Process audio features
  // Run inference
  // Return prediction
}
```

## 🎨 Design Features

- **Gradient Backgrounds**: Modern blue/purple gradients
- **Smooth Animations**: Hover states and transitions
- **Responsive Design**: Mobile-first approach
- **Glass Morphism**: Backdrop blur effects
- **Visual Feedback**: Recording states and loading indicators

## 🛠 Technologies Used

**Frontend:**
- React 18 + TypeScript
- Vite for development
- Tailwind CSS for styling
- React Router for navigation
- Lucide React for icons

**Backend:**
- Node.js + Express
- Multer for file uploads
- CORS for cross-origin requests
- install ffmpeg 8.0 full build
- 
**MODEL integration:**
- run the provided .ipynb file in colab 
- use the url provided in the last cell 
- replace it in the /backend/emotion_detection.py url 
- keep the last cell running 

## 📱 Browser Support

- Chrome/Edge: Full support
- Firefox: Full support
- Safari: Full support
- Mobile browsers: Full support

**Note:** Requires microphone access permission for audio recording.

## 🔒 Privacy

- No audio data is stored permanently
- All processing happens locally/server-side only
- No user accounts or tracking required

## 🚀 Deployment Ready

Both frontend and backend are configured for easy deployment to cloud platforms like Vercel, Netlify, or Railway.