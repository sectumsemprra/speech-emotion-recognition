import React from 'react';
import { User, Users, AlertCircle, Mic, Waves, BarChart3 } from 'lucide-react';
import { GenderData } from '../pages/DetectGender';

interface GenderResultProps {
  data: GenderData;
}

const GenderResult: React.FC<GenderResultProps> = ({ data }) => {
  const getGenderIcon = (gender: string) => {
    const genderLower = gender.toLowerCase();
    switch (genderLower) {
      case 'male':
        return <User className="w-12 h-12 text-blue-400" />;
      case 'female':
        return <Users className="w-12 h-12 text-pink-400" />;
      case 'unknown':
        return <Mic className="w-12 h-12 text-gray-400" />;
      case 'error':
        return <AlertCircle className="w-12 h-12 text-red-400" />;
      default:
        return <User className="w-12 h-12 text-purple-400" />;
    }
  };

  const getGenderColor = (gender: string) => {
    const genderLower = gender.toLowerCase();
    switch (genderLower) {
      case 'male':
        return 'from-blue-400 to-blue-600';
      case 'female':
        return 'from-pink-400 to-pink-600';
      case 'unknown':
        return 'from-gray-400 to-gray-600';
      case 'error':
        return 'from-red-400 to-red-600';
      default:
        return 'from-purple-400 to-indigo-600';
    }
  };

  const formatFrequency = (freq: number) => {
    return freq > 0 ? `${freq.toFixed(0)} Hz` : 'N/A';
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
      </div>

      {/* Probability Breakdown */}
      <div>
        <h4 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
          <BarChart3 className="w-5 h-5" />
          Probability Breakdown
        </h4>
        <div className="space-y-3">
          <div className="flex items-center justify-between bg-white/5 rounded-lg p-3">
            <div className="flex items-center gap-3">
              <User className="w-6 h-6 text-blue-400" />
              <span className="text-white">Male</span>
            </div>
            <div className="flex items-center gap-3">
              <div className="w-24 bg-white/10 rounded-full h-2">
                <div 
                  className="bg-gradient-to-r from-blue-400 to-blue-600 h-2 rounded-full transition-all duration-500"
                  style={{ width: `${data.probabilities.male * 100}%` }}
                ></div>
              </div>
              <span className="text-blue-100 text-sm w-12 text-right">
                {(data.probabilities.male * 100).toFixed(1)}%
              </span>
            </div>
          </div>

          <div className="flex items-center justify-between bg-white/5 rounded-lg p-3">
            <div className="flex items-center gap-3">
              <Users className="w-6 h-6 text-pink-400" />
              <span className="text-white">Female</span>
            </div>
            <div className="flex items-center gap-3">
              <div className="w-24 bg-white/10 rounded-full h-2">
                <div 
                  className="bg-gradient-to-r from-pink-400 to-pink-600 h-2 rounded-full transition-all duration-500"
                  style={{ width: `${data.probabilities.female * 100}%` }}
                ></div>
              </div>
              <span className="text-blue-100 text-sm w-12 text-right">
                {(data.probabilities.female * 100).toFixed(1)}%
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Technical Details */}
      <div>
        <h4 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
          <Waves className="w-5 h-5" />
          Voice Analysis Features
        </h4>
        <div className="grid grid-cols-2 gap-3">
          <div className="bg-white/5 rounded-lg p-4">
            <div className="text-sm text-blue-200">Fundamental Frequency</div>
            <div className="text-lg font-semibold text-white">
              {formatFrequency(data.features_used.f0_mean)}
            </div>
            <div className="text-xs text-blue-300 mt-1">
              Pitch of voice (key indicator)
            </div>
          </div>
          
          <div className="bg-white/5 rounded-lg p-4">
            <div className="text-sm text-blue-200">Spectral Centroid</div>
            <div className="text-lg font-semibold text-white">
              {formatFrequency(data.features_used.spectral_centroid)}
            </div>
            <div className="text-xs text-blue-300 mt-1">
              Voice brightness measure
            </div>
          </div>
          
          <div className="bg-white/5 rounded-lg p-4">
            <div className="text-sm text-blue-200">First Formant (F1)</div>
            <div className="text-lg font-semibold text-white">
              {formatFrequency(data.features_used.f1_approx)}
            </div>
            <div className="text-xs text-blue-300 mt-1">
              Vocal tract resonance
            </div>
          </div>
          
          <div className="bg-white/5 rounded-lg p-4">
            <div className="text-sm text-blue-200">Second Formant (F2)</div>
            <div className="text-lg font-semibold text-white">
              {formatFrequency(data.features_used.f2_approx)}
            </div>
            <div className="text-xs text-blue-300 mt-1">
              Vowel characteristics
            </div>
          </div>
        </div>
      </div>

      {/* Interpretation Guide */}
      <div className="bg-white/3 rounded-lg p-4">
        <h5 className="text-sm font-semibold text-white mb-2">Quick Interpretation:</h5>
        <div className="text-xs text-blue-200 space-y-1">
          <p>• <strong>F0 &lt; 150Hz:</strong> Typically male voice</p>
          <p>• <strong>F0 &gt; 200Hz:</strong> Typically female voice</p>
          <p>• <strong>Higher spectral centroid:</strong> Brighter voice (often female)</p>
          <p>• <strong>Higher formants:</strong> Smaller vocal tract (often female)</p>
        </div>
      </div>
    </div>
  );
};

export default GenderResult;