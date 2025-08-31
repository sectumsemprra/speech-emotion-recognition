import React, { useState, useRef } from 'react';
import { Mic, MicOff, Square } from 'lucide-react';

interface AudioRecorderProps {
onAudioReady: (audioBlob: Blob) => void;
}

const AudioRecorder: React.FC<AudioRecorderProps> = ({ onAudioReady }) => {
const [isRecording, setIsRecording] = useState(false);
const [hasRecorded, setHasRecorded] = useState(false);
const [recordingTime, setRecordingTime] = useState(0);
const mediaRecorderRef = useRef<MediaRecorder | null>(null);
const audioChunksRef = useRef<Blob[]>([]);
const timerRef = useRef<number | null>(null);
const streamRef = useRef<MediaStream | null>(null);

const startRecording = async () => {
try {
const stream = await navigator.mediaDevices.getUserMedia({
audio: {
echoCancellation: true,
noiseSuppression: true,
autoGainControl: true,
sampleRate: 44100,
sampleSize: 16,
channelCount: 1 // Force mono
}
});

streamRef.current = stream;

// Use webm format with opus codec for better quality
let mimeType = 'audio/webm;codecs=opus';

// Fallback to other formats if webm is not supported
if (!MediaRecorder.isTypeSupported(mimeType)) {
if (MediaRecorder.isTypeSupported('audio/webm')) {
mimeType = 'audio/webm';
} else if (MediaRecorder.isTypeSupported('audio/mp4')) {
mimeType = 'audio/mp4';
} else {
mimeType = 'audio/wav';
}
}

mediaRecorderRef.current = new MediaRecorder(stream, {
mimeType: mimeType,
audioBitsPerSecond: 128000 // 128 kbps
});

audioChunksRef.current = [];
setRecordingTime(0);

mediaRecorderRef.current.ondataavailable = (event) => {
if (event.data.size > 0) {
audioChunksRef.current.push(event.data);
}
};

mediaRecorderRef.current.onstop = async () => {
const audioBlob = new Blob(audioChunksRef.current, { type: mimeType });

console.log(`Recorded ${audioChunksRef.current.length} chunks, total size: ${audioBlob.size} bytes`);
console.log(`MIME type: ${mimeType}`);

onAudioReady(audioBlob);
setHasRecorded(true);

// Clean up
if (streamRef.current) {
streamRef.current.getTracks().forEach(track => track.stop());
streamRef.current = null;
}

if (timerRef.current) {
clearInterval(timerRef.current);
timerRef.current = null;
}
};

mediaRecorderRef.current.onerror = (event) => {
console.error('MediaRecorder error:', event);
alert('Recording failed. Please try again.');
stopRecording();
};

// Start recording with data collection every 100ms for smoother experience
mediaRecorderRef.current.start(100);
setIsRecording(true);

// Start timer
timerRef.current = setInterval(() => {
setRecordingTime(prev => prev + 1);
}, 1000);

} catch (error) {
console.error('Error starting recording:', error);
let errorMessage = 'Could not access microphone. ';

if (error instanceof DOMException) {
switch (error.name) {
case 'NotAllowedError':
errorMessage += 'Please allow microphone access and try again.';
break;
case 'NotFoundError':
errorMessage += 'No microphone found. Please connect a microphone and try again.';
break;
case 'NotReadableError':
errorMessage += 'Microphone is already in use by another application.';
break;
default:
errorMessage += 'Please check your microphone settings and try again.';
}
} else {
errorMessage += 'Please check your microphone settings and try again.';
}

alert(errorMessage);
}
};

const stopRecording = () => {
if (mediaRecorderRef.current && isRecording) {
mediaRecorderRef.current.stop();
setIsRecording(false);

if (timerRef.current) {
clearInterval(timerRef.current);
timerRef.current = null;
}
}
};

const formatTime = (seconds: number) => {
const mins = Math.floor(seconds / 60);
const secs = seconds % 60;
return `${mins}:${secs.toString().padStart(2, '0')}`;
};

const resetRecorder = () => {
setHasRecorded(false);
setRecordingTime(0);
};

return (
<div className="text-center">
{/* Recording Status */}
{isRecording && (
<div className="mb-6">
<div className="flex items-center justify-center gap-3 text-red-400 mb-2">
<div className="w-3 h-3 bg-red-500 rounded-full animate-pulse"></div>
<span className="font-medium">Recording...</span>
<span className="text-white font-mono">{formatTime(recordingTime)}</span>
</div>
<div className="w-full bg-white/10 rounded-full h-2">
<div className="bg-gradient-to-r from-red-400 to-red-600 h-2 rounded-full animate-pulse"></div>
</div>

{/* Recording guidelines */}
<div className="mt-4 text-sm text-blue-200/80">
<p>💡 Speak clearly for better emotion detection</p>
<p>⏱️ Recommended: 5-30 seconds</p>
</div>
</div>
)}

{/* Recording Button */}
<div className="mb-6">
{!isRecording ? (
<button
onClick={startRecording}
disabled={isRecording}
className="group relative bg-gradient-to-r from-blue-500 to-purple-600 hover:from-blue-600 hover:to-purple-700 disabled:from-gray-500 disabled:to-gray-600 text-white rounded-full p-8 shadow-2xl hover:shadow-blue-500/30 transform hover:scale-105 disabled:scale-100 transition-all duration-200 disabled:cursor-not-allowed"
>
<div className="absolute inset-0 bg-gradient-to-r from-blue-400 to-purple-500 rounded-full blur-xl opacity-30 group-hover:opacity-50 group-disabled:opacity-10 transition-opacity"></div>
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
<div className="space-y-2">
<p className="text-blue-100">Speak naturally, then click stop when finished</p>
<p className="text-sm text-blue-200/70">Recording in high quality for accurate analysis</p>
</div>
)}
{!isRecording && hasRecorded && (
<div className="space-y-2">
<p className="text-green-300">✅ Recording complete! Processing emotion...</p>
<button
onClick={resetRecorder}
className="text-sm text-blue-300 hover:text-white transition-colors underline"
>
Record again
</button>
</div>
)}
</div>

{/* Browser Compatibility Info */}
<div className="mt-6 text-xs text-blue-200/70 max-w-md mx-auto">
<p>🔒 Audio is processed locally and securely</p>
<p>🎙️ Works best with Chrome, Firefox, or Safari</p>
<p>📱 Mobile recording supported</p>
</div>

{/* Audio Format Info for Debugging */}
{import.meta.env?.MODE === 'development' && (
<div className="mt-4 text-xs text-gray-400">
<p>Supported formats:</p>
<p>WebM Opus: {MediaRecorder.isTypeSupported('audio/webm;codecs=opus') ? '✅' : '❌'}</p>
<p>WebM: {MediaRecorder.isTypeSupported('audio/webm') ? '✅' : '❌'}</p>
<p>MP4: {MediaRecorder.isTypeSupported('audio/mp4') ? '✅' : '❌'}</p>
</div>
)}
</div>
);
};

export default AudioRecorder;
