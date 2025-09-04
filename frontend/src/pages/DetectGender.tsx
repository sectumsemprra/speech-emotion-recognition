import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import AudioRecorder from '../components/AudioRecorder';
import GenderResult from '../components/GenderResult';
import { ArrowLeft, User } from 'lucide-react';

export interface GenderData {
  gender: string;
  confidence: number;
  method?: string;
  scores: {
    male_score: number;
    female_score: number;
  };
  feature_analysis: {
    f0_hz: number;
    spectral_centroid_hz: number;
    f1_hz: number;
    f2_hz: number;
    feature_votes?: any;
    feature_confidences?: any;
  };
  all_features: Record<string, number>;
}

const DetectGender = () => {
  const [genderResult, setGenderResult] = useState<GenderData | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  const handleAudioUpload = async (audioBlob: Blob) => {
    setIsAnalyzing(true);
    setGenderResult(null);

    try {
      const formData = new FormData();
      formData.append('audio', audioBlob, 'recording.wav');

      const response = await fetch('http://localhost:5001/predict', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      setGenderResult(result);
    } catch (error) {
      console.error('Error uploading audio:', error);
      // Show fallback result for demo
      setGenderResult({
        gender: "Error",
        confidence: 0,
        probabilities: {
          male: 0.5,
          female: 0.5
        },
        features_used: {
          f0_mean: 0,
          spectral_centroid: 0,
          f1_approx: 0,
          f2_approx: 0
        }
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
            <User className="w-6 h-6" />
            <span className="text-lg font-semibold">Gender Classification</span>
          </div>
        </div>

        <div className="grid lg:grid-cols-2 gap-8 items-start">
          {/* Recording Section */}
          <div className="bg-white/5 backdrop-blur-sm rounded-3xl p-8 border border-white/10">
            <h2 className="text-2xl font-bold text-white mb-6">Record Your Voice</h2>
            <p className="text-blue-100 mb-8">
              Click the record button and speak naturally. We'll analyze your voice using DSP techniques to classify gender.
            </p>
            <AudioRecorder onAudioReady={handleAudioUpload} />
          </div>

          {/* Results Section */}
          <div className="bg-white/5 backdrop-blur-sm rounded-3xl p-8 border border-white/10">
            <h2 className="text-2xl font-bold text-white mb-6">Gender Analysis</h2>
            
            {isAnalyzing && (
              <div className="text-center py-12">
                <div className="inline-flex items-center gap-3 text-blue-200">
                  <div className="animate-spin rounded-full h-6 w-6 border-2 border-blue-400 border-t-transparent"></div>
                  Analyzing voice characteristics...
                </div>
              </div>
            )}

            {!isAnalyzing && !genderResult && (
              <div className="text-center py-12 text-blue-200">
                Record audio to see gender classification results here
              </div>
            )}

            {!isAnalyzing && genderResult && (
              <GenderResult data={genderResult} />
            )}
          </div>
        </div>

        {/* Technical Info */}
        <div className="mt-8 bg-white/5 backdrop-blur-sm rounded-2xl p-6 border border-white/10">
          <h3 className="text-lg font-semibold text-white mb-4">How It Works</h3>
          <div className="grid md:grid-cols-2 gap-6 text-sm text-blue-100">
            <div>
              <h4 className="font-medium text-white mb-2">DSP Techniques Used:</h4>
              <ul className="space-y-1">
                <li>• Fundamental Frequency (F0) Analysis</li>
                <li>• Spectral Centroid Calculation</li>
                <li>• Formant Frequency Estimation</li>
                <li>• MFCC Feature Extraction</li>
              </ul>
            </div>
            <div>
              <h4 className="font-medium text-white mb-2">Classification Method:</h4>
              <ul className="space-y-1">
                <li>• Rule-based approach using voice characteristics</li>
                <li>• No ML models - pure signal processing</li>
                <li>• Focuses on pitch and spectral features</li>
                <li>• Real-time analysis</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default DetectGender;