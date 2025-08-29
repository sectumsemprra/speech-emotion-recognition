import React, { useState, useRef } from 'react';
import { Mic, MicOff, Square } from 'lucide-react';

interface AudioRecorderProps {
  onAudioReady: (audioBlob: Blob) => void;
}

const AudioRecorder: React.FC<AudioRecorderProps> = ({ onAudioReady }) => {
  const [isRecording, setIsRecording] = useState(false);
  const [hasRecorded, setHasRecorded] = useState(false);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          sampleRate: 44100
        } 
      });
      
      mediaRecorderRef.current = new MediaRecorder(stream, {
        mimeType: 'audio/webm'
      });
      
      audioChunksRef.current = [];
      
      mediaRecorderRef.current.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };
      
      mediaRecorderRef.current.onstop = async () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
        
        // Convert to WAV format (simplified for demo)
        const wavBlob = new Blob([audioBlob], { type: 'audio/wav' });
        
        onAudioReady(wavBlob);
        setHasRecorded(true);
        
        // Clean up
        stream.getTracks().forEach(track => track.stop());
      };
      
      mediaRecorderRef.current.start(1000);
      setIsRecording(true);
    } catch (error) {
      console.error('Error starting recording:', error);
      alert('Could not access microphone. Please check permissions.');
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  };

  return (
    <div className="text-center">
      {/* Recording Status */}
      {isRecording && (
        <div className="mb-6">
          <div className="flex items-center justify-center gap-3 text-red-400 mb-2">
            <div className="w-3 h-3 bg-red-500 rounded-full animate-pulse"></div>
            <span className="font-medium">Recording...</span>
          </div>
          <div className="w-full bg-white/10 rounded-full h-2">
            <div className="bg-gradient-to-r from-red-400 to-red-600 h-2 rounded-full animate-pulse"></div>
          </div>
        </div>
      )}

      {/* Recording Button */}
      <div className="mb-6">
        {!isRecording ? (
          <button
            onClick={startRecording}
            className="group relative bg-gradient-to-r from-blue-500 to-purple-600 hover:from-blue-600 hover:to-purple-700 text-white rounded-full p-8 shadow-2xl hover:shadow-blue-500/30 transform hover:scale-105 transition-all duration-200"
          >
            <div className="absolute inset-0 bg-gradient-to-r from-blue-400 to-purple-500 rounded-full blur-xl opacity-30 group-hover:opacity-50 transition-opacity"></div>
            <Mic className="w-12 h-12 relative z-10" />
          </button>
        ) : (
          <button
            onClick={stopRecording}
            className="group relative bg-gradient-to-r from-red-500 to-red-600 hover:from-red-600 hover:to-red-700 text-white rounded-full p-8 shadow-2xl hover:shadow-red-500/30 transform hover:scale-105 transition-all duration-200"
          >
            <div className="absolute inset-0 bg-gradient-to-r from-red-400 to-red-500 rounded-full blur-xl opacity-30 group-hover:opacity-50 transition-opacity"></div>
            <Square className="w-12 h-12 relative z-10" />
          </button>
        )}
      </div>

      {/* Instructions */}
      <div className="text-center">
        {!isRecording && !hasRecorded && (
          <p className="text-blue-100">Click the microphone to start recording</p>
        )}
        {isRecording && (
          <p className="text-blue-100">Speak naturally, then click stop when finished</p>
        )}
        {!isRecording && hasRecorded && (
          <p className="text-green-300">✅ Recording complete! Processing emotion...</p>
        )}
      </div>

      {/* Permissions Note */}
      <div className="mt-6 text-xs text-blue-200/70 max-w-md mx-auto">
        <p>📝 This app requires microphone access to record your voice for emotion analysis.</p>
      </div>
    </div>
  );
};

export default AudioRecorder;