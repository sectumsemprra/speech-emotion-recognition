import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import AudioRecorder from '../components/AudioRecorder';
import EmotionResult from '../components/EmotionResult';
import { ArrowLeft, Brain } from 'lucide-react';

export interface EmotionData {
  emotion: string;
  confidence: number;
  topEmotions: Array<{
    emotion: string;
    score: number;
  }>;
}

const DetectEmotion = () => {
  const [emotionResult, setEmotionResult] = useState<EmotionData | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  const handleAudioUpload = async (audioBlob: Blob) => {
    setIsAnalyzing(true);
    setEmotionResult(null);

    try {
      const formData = new FormData();
      formData.append('audio', audioBlob, 'recording.wav');

      const response = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      setEmotionResult(result);
    } catch (error) {
      console.error('Error uploading audio:', error);
      // Show fallback result for demo
      setEmotionResult({
        emotion: "Error",
        confidence: 0,
        topEmotions: [
          { emotion: "Error", score: 1.0 },
          { emotion: "Please start backend", score: 0.0 },
          { emotion: "See README", score: 0.0 }
        ]
      });
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <div className="min-h-screen p-8">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <Link
            to="/"
            className="inline-flex items-center gap-2 text-blue-200 hover:text-white transition-colors duration-200"
          >
            <ArrowLeft className="w-5 h-5" />
            Back to Home
          </Link>
          
          <div className="flex items-center gap-3 text-white">
            <Brain className="w-6 h-6" />
            <span className="text-lg font-semibold">Emotion Detection</span>
          </div>
        </div>

        <div className="grid lg:grid-cols-2 gap-8 items-start">
          {/* Recording Section */}
          <div className="bg-white/5 backdrop-blur-sm rounded-3xl p-8 border border-white/10">
            <h2 className="text-2xl font-bold text-white mb-6">Record Your Voice</h2>
            <p className="text-blue-100 mb-8">
              Click the record button and speak naturally. We'll analyze the emotional content of your voice.
            </p>
            <AudioRecorder onAudioReady={handleAudioUpload} />
          </div>

          {/* Results Section */}
          <div className="bg-white/5 backdrop-blur-sm rounded-3xl p-8 border border-white/10">
            <h2 className="text-2xl font-bold text-white mb-6">Emotion Analysis</h2>
            
            {isAnalyzing && (
              <div className="text-center py-12">
                <div className="inline-flex items-center gap-3 text-blue-200">
                  <div className="animate-spin rounded-full h-6 w-6 border-2 border-blue-400 border-t-transparent"></div>
                  Analyzing emotions...
                </div>
              </div>
            )}

            {!isAnalyzing && !emotionResult && (
              <div className="text-center py-12 text-blue-200">
                Record audio to see emotion analysis results here
              </div>
            )}

            {!isAnalyzing && emotionResult && (
              <EmotionResult data={emotionResult} />
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default DetectEmotion;