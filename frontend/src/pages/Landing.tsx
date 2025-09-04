import React from 'react';
import { Link } from 'react-router-dom';
import { Mic, Waves, Brain, ArrowRight, User } from 'lucide-react';

const Landing = () => {
  return (
    <div className="min-h-screen flex items-center justify-center p-8">
      <div className="max-w-4xl mx-auto text-center">
        {/* Hero Section */}
        <div className="mb-12">
          <div className="flex justify-center mb-6">
            <div className="relative">
              <div className="absolute inset-0 bg-gradient-to-r from-blue-500 to-purple-600 rounded-full blur-xl opacity-30 animate-pulse"></div>
              <div className="relative bg-white/10 backdrop-blur-sm rounded-full p-6">
                <Brain className="w-16 h-16 text-white" />
              </div>
            </div>
          </div>
          
          <h1 className="text-5xl md:text-7xl font-bold text-white mb-6 leading-tight">
            Speech
            <span className="block bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
              Analysis
            </span>
          </h1>
          
          <p className="text-xl md:text-2xl text-blue-100 mb-8 max-w-2xl mx-auto leading-relaxed">
            Analyze your voice using advanced AI and DSP technology. 
            Detect emotions and classify gender from speech patterns.
          </p>
        </div>

        {/* Features */}
        <div className="grid md:grid-cols-3 gap-8 mb-12">
          <div className="bg-white/5 backdrop-blur-sm rounded-2xl p-6 border border-white/10">
            <Mic className="w-8 h-8 text-blue-400 mb-4 mx-auto" />
            <h3 className="text-lg font-semibold text-white mb-2">Record Audio</h3>
            <p className="text-blue-100 text-sm">Simple one-click recording with automatic format conversion</p>
          </div>
          
          <div className="bg-white/5 backdrop-blur-sm rounded-2xl p-6 border border-white/10">
            <Brain className="w-8 h-8 text-purple-400 mb-4 mx-auto" />
            <h3 className="text-lg font-semibold text-white mb-2">Emotion Detection</h3>
            <p className="text-blue-100 text-sm">ML-powered emotion analysis from voice patterns</p>
          </div>
          
          <div className="bg-white/5 backdrop-blur-sm rounded-2xl p-6 border border-white/10">
            <User className="w-8 h-8 text-indigo-400 mb-4 mx-auto" />
            <h3 className="text-lg font-semibold text-white mb-2">Gender Classification</h3>
            <p className="text-blue-100 text-sm">DSP-based gender detection using voice characteristics</p>
          </div>
        </div>

        {/* CTA Buttons */}
        <div className="flex flex-col sm:flex-row gap-4 justify-center">
          <Link
            to="/detect"
            className="inline-flex items-center gap-3 bg-gradient-to-r from-blue-500 to-purple-600 text-white px-8 py-4 rounded-2xl font-semibold text-lg hover:from-blue-600 hover:to-purple-700 transform hover:scale-105 transition-all duration-200 shadow-2xl hover:shadow-blue-500/25"
          >
            <Brain className="w-5 h-5" />
            Emotion Detection
          </Link>
          
          <Link
            to="/gender"
            className="inline-flex items-center gap-3 bg-gradient-to-r from-pink-500 to-indigo-600 text-white px-8 py-4 rounded-2xl font-semibold text-lg hover:from-pink-600 hover:to-indigo-700 transform hover:scale-105 transition-all duration-200 shadow-2xl hover:shadow-pink-500/25"
          >
            <User className="w-5 h-5" />
            Gender Classification
          </Link>
        </div>

        <p className="text-blue-200 text-sm mt-6">No signup required • Free to use • Privacy focused</p>
      </div>
    </div>
  );
};

export default Landing;