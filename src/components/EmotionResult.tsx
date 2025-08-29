import React from 'react';
import { Heart, Frown, Smile, Meh, AlertCircle, Angry } from 'lucide-react';
import { EmotionData } from '../pages/DetectEmotion';

interface EmotionResultProps {
  data: EmotionData;
}

const EmotionResult: React.FC<EmotionResultProps> = ({ data }) => {
  const getEmotionIcon = (emotion: string) => {
    const emotionLower = emotion.toLowerCase();
    switch (emotionLower) {
      case 'happy':
      case 'joy':
        return <Smile className="w-12 h-12 text-yellow-400" />;
      case 'sad':
      case 'sadness':
        return <Frown className="w-12 h-12 text-blue-400" />;
      case 'angry':
      case 'anger':
        return <Angry className="w-12 h-12 text-red-400" />;
      case 'love':
      case 'affection':
        return <Heart className="w-12 h-12 text-pink-400" />;
      case 'neutral':
        return <Meh className="w-12 h-12 text-gray-400" />;
      case 'error':
        return <AlertCircle className="w-12 h-12 text-red-400" />;
      default:
        return <Brain className="w-12 h-12 text-purple-400" />;
    }
  };

  const getEmotionColor = (emotion: string) => {
    const emotionLower = emotion.toLowerCase();
    switch (emotionLower) {
      case 'happy':
      case 'joy':
        return 'from-yellow-400 to-orange-500';
      case 'sad':
      case 'sadness':
        return 'from-blue-400 to-indigo-600';
      case 'angry':
      case 'anger':
        return 'from-red-400 to-red-600';
      case 'love':
      case 'affection':
        return 'from-pink-400 to-rose-500';
      case 'neutral':
        return 'from-gray-400 to-gray-600';
      case 'error':
        return 'from-red-400 to-red-600';
      default:
        return 'from-purple-400 to-indigo-600';
    }
  };

  return (
    <div className="space-y-6">
      {/* Main Emotion Card */}
      <div className="text-center">
        <div className="relative mb-6">
          <div className={`absolute inset-0 bg-gradient-to-r ${getEmotionColor(data.emotion)} rounded-full blur-xl opacity-20 animate-pulse`}></div>
          <div className="relative bg-white/10 backdrop-blur-sm rounded-full p-6 inline-block">
            {getEmotionIcon(data.emotion)}
          </div>
        </div>
        
        <h3 className="text-3xl font-bold text-white mb-2 capitalize">
          {data.emotion}
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
            className={`bg-gradient-to-r ${getEmotionColor(data.emotion)} h-3 rounded-full transition-all duration-1000 ease-out`}
            style={{ width: `${data.confidence * 100}%` }}
          ></div>
        </div>
      </div>

      {/* Top Emotions Breakdown */}
      {data.topEmotions && data.topEmotions.length > 0 && (
        <div>
          <h4 className="text-lg font-semibold text-white mb-4">Emotion Breakdown</h4>
          <div className="space-y-3">
            {data.topEmotions.slice(0, 3).map((item, index) => (
              <div key={index} className="flex items-center justify-between bg-white/5 rounded-lg p-3">
                <div className="flex items-center gap-3">
                  <div className="w-8 h-8">
                    {getEmotionIcon(item.emotion)}
                  </div>
                  <span className="text-white capitalize">{item.emotion}</span>
                </div>
                <div className="flex items-center gap-3">
                  <div className="w-24 bg-white/10 rounded-full h-2">
                    <div 
                      className={`bg-gradient-to-r ${getEmotionColor(item.emotion)} h-2 rounded-full transition-all duration-500`}
                      style={{ width: `${item.score * 100}%` }}
                    ></div>
                  </div>
                  <span className="text-blue-100 text-sm w-12 text-right">
                    {(item.score * 100).toFixed(0)}%
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default EmotionResult;