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

### Backend (FastAPI + Python)
```bash
# Navigate to backend directory
cd backend

# Install Python dependencies
pip install -r requirements.txt

# Start the FastAPI server
python emotion_service.py
```

The backend API will run on `http://localhost:5000`

## 📁 Project Structure

```
├── frontend/
│   ├── src/
│   │   ├── pages/
│   │   │   ├── Landing.tsx          # Landing page with hero section
│   │   │   └── DetectEmotion.tsx    # Main emotion detection interface
│   │   ├── components/
│   │   │   ├── AudioRecorder.tsx    # Audio recording component
│   │   │   └── EmotionResult.tsx    # Emotion results display
│   │   └── App.tsx                  # Main app with routing
│   └── package.json                 # Frontend dependencies
├── backend/
│   ├── emotion_service.py          # FastAPI emotion detection service
│   ├── local_emotion_detection.py  # Standalone emotion detection script
│   └── requirements.txt            # Python dependencies
└── README.md                       # This file
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

### `GET /info`
Get service information.

### `GET /docs`
FastAPI automatic documentation (Swagger UI).

## 🔬 Machine Learning Integration

The backend now uses the **emotion2vec_plus_large** model from FunASR for real emotion detection. The model is loaded automatically when the service starts.

**Model Details:**
- **Model**: emotion2vec_plus_large from FunASR
- **Capabilities**: Multi-class emotion recognition from speech
- **Input**: Audio files (WAV, MP3, etc.)
- **Output**: Emotion classification with confidence scores

**To use a different model:**
1. Modify `emotion_service.py` 
2. Replace the model loading in the `load_model()` function
3. Update the prediction logic in `detect_emotion_from_file()`

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
- FastAPI + Python
- FunASR emotion2vec_plus_large model
- Automatic API documentation
- CORS enabled for frontend integration 

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