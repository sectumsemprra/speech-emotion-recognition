import React from 'react';
import { User, UserCheck, Activity, BarChart3, Zap, Waves } from 'lucide-react';

export interface GenderData {
  gender: string;
  confidence: number;
  method: string;
  feature_analysis?: {
    f0_hz: number;
    f1_hz: number;
    f2_hz: number;
    spectral_centroid_hz: number;
    feature_votes: Record<string, string>;
    feature_confidences: Record<string, number>;
  };
  scores: {
    male_score: number;
    female_score: number;
  };
  all_features: Record<string, number>;
}

interface GenderResultProps {
  data: GenderData;
}

const GenderResult: React.FC<GenderResultProps> = ({ data }) => {
  const getGenderIcon = (gender: string) => {
    switch (gender.toLowerCase()) {
      case 'male':
        return <User className="w-12 h-12 text-blue-400" />;
      case 'female':
        return <UserCheck className="w-12 h-12 text-pink-400" />;
      default:
        return <User className="w-12 h-12 text-gray-400" />;
    }
  };

  const getGenderColor = (gender: string) => {
    switch (gender.toLowerCase()) {
      case 'male':
        return 'from-blue-400 to-blue-600';
      case 'female':
        return 'from-pink-400 to-pink-600';
      default:
        return 'from-gray-400 to-gray-600';
    }
  };

  const getFeatureIcon = (feature: string) => {
    switch (feature) {
      case 'f0':
        return <Waves className="w-5 h-5" />;
      case 'f1':
      case 'f2':
        return <Activity className="w-5 h-5" />;
      case 'spectral':
        return <BarChart3 className="w-5 h-5" />;
      default:
        return <Zap className="w-5 h-5" />;
    }
  };

  const getFeatureName = (feature: string) => {
    switch (feature) {
      case 'f0':
        return 'Fundamental Frequency (F0)';
      case 'f1':
        return 'First Formant (F1)';
      case 'f2':
        return 'Second Formant (F2)';
      case 'spectral':
        return 'Spectral Centroid';
      default:
        return feature;
    }
  };

  const thresholds = {
    f0: { value: 165, unit: 'Hz', description: 'Voice pitch' },
    f1: { value: 730, unit: 'Hz', description: 'Vowel resonance 1' },
    f2: { value: 1090, unit: 'Hz', description: 'Vowel resonance 2' },
    spectral: { value: 2000, unit: 'Hz', description: 'Spectral brightness' }
  };

  const getFeatureValue = (feature: string): number => {
    if (!data.feature_analysis) return 0;
    switch (feature) {
      case 'f0':
        return data.feature_analysis.f0_hz;
      case 'f1':
        return data.feature_analysis.f1_hz;
      case 'f2':
        return data.feature_analysis.f2_hz;
      case 'spectral':
        return data.feature_analysis.spectral_centroid_hz;
      default:
        return 0;
    }
  };

  return (
    <div className="space-y-6">
      {/* Main Gender Card */}
      <div className="text-center">
        <div className="relative mb-6">
          <div className={`absolute inset-0 bg-gradient-to-r ${getGenderColor(data.gender)} rounded-full blur-xl opacity-20 animate-pulse`}></div>
          <div className="relative bg-white/10 backdrop-blur-sm rounded-full p-6 inline-block">
            {getGenderIcon(data.gender)}
          </div>
        </div>
        
        <h3 className="text-3xl font-bold text-white mb-2 capitalize">
          {data.gender}
        </h3>
        
        <div className="text-blue-100 mb-4">
          <span className="text-lg">Confidence: </span>
          <span className="text-xl font-semibold text-white">
            {(data.confidence * 100).toFixed(1)}%
          </span>
        </div>

        {/* Confidence Bar */}
        <div className="w-full bg-white/10 rounded-full h-3 mb-6">
          <div 
            className={`bg-gradient-to-r ${getGenderColor(data.gender)} h-3 rounded-full transition-all duration-1000 ease-out`}
            style={{ width: `${data.confidence * 100}%` }}
          ></div>
        </div>

        {/* Method Badge */}
        <div className="inline-flex items-center gap-2 bg-white/10 rounded-full px-4 py-2 text-sm text-blue-200">
          <Activity className="w-4 h-4" />
          Method: {data.method || 'DSP Analysis'}
        </div>
      </div>

      {/* Gender Scores */}
      <div className="bg-white/5 rounded-xl p-4">
        <h4 className="text-lg font-semibold text-white mb-4">Classification Scores</h4>
        <div className="grid grid-cols-2 gap-4">
          <div className="text-center">
            <div className="text-2xl font-bold text-blue-400">
              {(data.scores.male_score * 100).toFixed(1)}%
            </div>
            <div className="text-blue-200 text-sm">Male Score</div>
            <div className="w-full bg-white/10 rounded-full h-2 mt-2">
              <div 
                className="bg-gradient-to-r from-blue-400 to-blue-600 h-2 rounded-full transition-all duration-500"
                style={{ width: `${data.scores.male_score * 100}%` }}
              ></div>
            </div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-pink-400">
              {(data.scores.female_score * 100).toFixed(1)}%
            </div>
            <div className="text-blue-200 text-sm">Female Score</div>
            <div className="w-full bg-white/10 rounded-full h-2 mt-2">
              <div 
                className="bg-gradient-to-r from-pink-400 to-pink-600 h-2 rounded-full transition-all duration-500"
                style={{ width: `${data.scores.female_score * 100}%` }}
              ></div>
            </div>
          </div>
        </div>
      </div>

      {/* DSP Features Analysis */}
      {data.feature_analysis && (
        <div className="bg-white/5 rounded-xl p-4">
          <h4 className="text-lg font-semibold text-white mb-4">DSP Feature Analysis</h4>
          <div className="space-y-4">
            {Object.entries(data.feature_analysis.feature_votes).map(([feature, vote]) => {
              const featureValue = getFeatureValue(feature);
              const threshold = thresholds[feature as keyof typeof thresholds];
              const confidence = data.feature_analysis?.feature_confidences[feature] || 0;
              
              if (!threshold || featureValue === 0) return null;

              return (
                <div key={feature} className="bg-white/5 rounded-lg p-4">
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center gap-3">
                      {getFeatureIcon(feature)}
                      <div>
                        <div className="font-medium text-white">
                          {getFeatureName(feature)}
                        </div>
                        <div className="text-xs text-blue-300">
                          {threshold.description}
                        </div>
                      </div>
                    </div>
                    <div className="text-right">
                      <div className={`font-semibold ${vote === 'male' ? 'text-blue-400' : 'text-pink-400'}`}>
                        {vote.toUpperCase()}
                      </div>
                      <div className="text-xs text-blue-200">
                        {(confidence * 100).toFixed(0)}% confident
                      </div>
                    </div>
                  </div>
                  
                  {/* Feature Value vs Threshold */}
                  <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span className="text-blue-200">
                        Measured: {featureValue.toFixed(1)} {threshold.unit}
                      </span>
                      <span className="text-blue-200">
                        Threshold: {threshold.value} {threshold.unit}
                      </span>
                    </div>
                    
                    {/* Visual threshold indicator */}
                    <div className="relative">
                      <div className="w-full bg-white/10 rounded-full h-3">
                        {/* Threshold line */}
                        <div 
                          className="absolute top-0 w-0.5 h-3 bg-yellow-400"
                          style={{ left: '50%' }}
                        ></div>
                        {/* Feature value indicator */}
                        <div 
                          className={`h-3 rounded-full transition-all duration-500 ${
                            featureValue > threshold.value 
                              ? 'bg-gradient-to-r from-pink-400 to-pink-600' 
                              : 'bg-gradient-to-r from-blue-400 to-blue-600'
                          }`}
                          style={{ 
                            width: `${Math.min(100, (featureValue / (threshold.value * 2)) * 100)}%` 
                          }}
                        ></div>
                      </div>
                      <div className="flex justify-between text-xs text-blue-300 mt-1">
                        <span>Male Range</span>
                        <span className="text-yellow-400">Threshold</span>
                        <span>Female Range</span>
                      </div>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Additional Features */}
      {data.all_features && Object.keys(data.all_features).length > 4 && (
        <div className="bg-white/5 rounded-xl p-4">
          <h4 className="text-lg font-semibold text-white mb-4">Additional DSP Features</h4>
          <div className="grid grid-cols-2 gap-4 text-sm">
            {Object.entries(data.all_features)
              .filter(([key]) => !['f0_mean', 'formant_f1_mean', 'formant_f2_mean', 'spectral_centroid'].includes(key))
              .slice(0, 8)
              .map(([key, value]) => (
                <div key={key} className="flex justify-between bg-white/5 rounded p-2">
                  <span className="text-blue-200 capitalize">
                    {key.replace(/_/g, ' ').replace(/([a-z])([A-Z])/g, '$1 $2')}
                  </span>
                  <span className="text-white font-mono">
                    {typeof value === 'number' ? value.toFixed(2) : value}
                  </span>
                </div>
              ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default GenderResult;

