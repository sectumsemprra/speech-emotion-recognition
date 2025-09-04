import { useState } from 'react';
import { Link } from 'react-router-dom';
import AudioRecorder from '../components/AudioRecorder';
import EmotionResult from '../components/EmotionResult';
import GenderResult, { GenderData } from '../components/GenderResult';
import { ArrowLeft, Brain, Users } from 'lucide-react';

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
  const [genderResult, setGenderResult] = useState<GenderData | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisType, setAnalysisType] = useState<'both' | 'emotion' | 'gender'>('both');

  const handleAudioUpload = async (audioBlob: Blob) => {
    setIsAnalyzing(true);
    setEmotionResult(null);
    setGenderResult(null);

    const baseUrl = 'http://localhost:5000';
    
    try {
      // Create separate FormData for each request to avoid cloning issues
      const promises = [];
      
      if (analysisType === 'emotion' || analysisType === 'both') {
        const emotionFormData = new FormData();
        emotionFormData.append('audio', audioBlob, 'recording.wav');
        
        promises.push(
          fetch(`${baseUrl}/predict`, {
            method: 'POST',
            body: emotionFormData,
          }).then(res => res.json()).then(data => ({ type: 'emotion', data }))
        );
      }
      
      if (analysisType === 'gender' || analysisType === 'both') {
        const genderFormData = new FormData();
        genderFormData.append('audio', audioBlob, 'recording.wav');
        
        promises.push(
          fetch(`${baseUrl}/classify-gender`, {
            method: 'POST',
            body: genderFormData,
          }).then(res => res.json()).then(data => ({ type: 'gender', data }))
        );
      }

      const results = await Promise.all(promises);
      
      results.forEach(result => {
        if (result.type === 'emotion') {
          setEmotionResult(result.data);
        } else if (result.type === 'gender') {
          setGenderResult(result.data);
        }
      });

    } catch (error) {
      console.error('Error uploading audio:', error);
      
      // Show fallback results for demo
      if (analysisType === 'emotion' || analysisType === 'both') {
        setEmotionResult({
          emotion: "Error",
          confidence: 0,
          topEmotions: [
            { emotion: "Error", score: 1.0 },
            { emotion: "Please start backend", score: 0.0 },
            { emotion: "See README", score: 0.0 }
          ]
        });
      }
      
      if (analysisType === 'gender' || analysisType === 'both') {
        setGenderResult({
          gender: "unknown",
          confidence: 0,
          method: "error",
          scores: { male_score: 0, female_score: 0 },
          all_features: {}
        });
      }
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
            <span className="text-lg font-semibold">Speech Analysis</span>
          </div>
        </div>

        {/* Analysis Type Selector */}
        <div className="bg-white/5 backdrop-blur-sm rounded-3xl p-6 border border-white/10 mb-8">
          <h2 className="text-xl font-bold text-white mb-4">Analysis Type</h2>
          <div className="flex gap-4 flex-wrap">
            {[
              { key: 'both', label: 'Both Emotion & Gender', icon: <Brain className="w-4 h-4" /> },
              { key: 'emotion', label: 'Emotion Only', icon: <Brain className="w-4 h-4" /> },
              { key: 'gender', label: 'Gender Only', icon: <Users className="w-4 h-4" /> }
            ].map(({ key, label, icon }) => (
              <button
                key={key}
                onClick={() => setAnalysisType(key as 'both' | 'emotion' | 'gender')}
                className={`flex items-center gap-2 px-4 py-2 rounded-xl transition-all duration-200 ${
                  analysisType === key
                    ? 'bg-blue-500 text-white shadow-lg'
                    : 'bg-white/10 text-blue-200 hover:bg-white/20'
                }`}
              >
                {icon}
                {label}
              </button>
            ))}
          </div>
        </div>

        <div className="grid lg:grid-cols-2 gap-8 items-start">
          {/* Recording Section */}
          <div className="bg-white/5 backdrop-blur-sm rounded-3xl p-8 border border-white/10">
            <h2 className="text-2xl font-bold text-white mb-6">Record Your Voice</h2>
            <p className="text-blue-100 mb-8">
              Click the record button and speak naturally. We'll analyze your speech using advanced DSP techniques.
            </p>
            <AudioRecorder onAudioReady={handleAudioUpload} />
          </div>

          {/* Results Section */}
          <div className="bg-white/5 backdrop-blur-sm rounded-3xl p-8 border border-white/10">
            <h2 className="text-2xl font-bold text-white mb-6">Analysis Results</h2>
            
            {isAnalyzing && (
              <div className="text-center py-12">
                <div className="inline-flex items-center gap-3 text-blue-200">
                  <div className="animate-spin rounded-full h-6 w-6 border-2 border-blue-400 border-t-transparent"></div>
                  Analyzing speech...
                </div>
              </div>
            )}

            {!isAnalyzing && !emotionResult && !genderResult && (
              <div className="text-center py-12 text-blue-200">
                Record audio to see analysis results here
              </div>
            )}

            {!isAnalyzing && (emotionResult || genderResult) && (
              <div className="space-y-8">
                {/* Emotion Results */}
                {emotionResult && (
                  <div>
                    <div className="flex items-center gap-2 mb-4">
                      <Brain className="w-5 h-5 text-blue-400" />
                      <h3 className="text-lg font-semibold text-white">Emotion Detection</h3>
                    </div>
                    <EmotionResult data={emotionResult} />
                  </div>
                )}

                {/* Gender Results */}
                {genderResult && (
                  <div>
                    <div className="flex items-center gap-2 mb-4">
                      <Users className="w-5 h-5 text-pink-400" />
                      <h3 className="text-lg font-semibold text-white">Gender Classification</h3>
                    </div>
                    <GenderResult data={genderResult} />
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default DetectEmotion;