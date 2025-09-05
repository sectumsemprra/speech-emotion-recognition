import { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import AudioRecorder from '../components/AudioRecorder';
import EmotionResult from '../components/EmotionResult';
import GenderResult from '../components/GenderResult';
import { ArrowLeft, Brain, Users } from 'lucide-react';
import { GenderData } from '../pages/DetectGender';

export interface EmotionData {
  emotion: string;
  confidence: number;
  topEmotions: Array<{
    emotion: string;
    score: number;
  }>;
}

/** ---- DSP types ---- */
type FilterPreset = {
  filter_type: 'bandpass' | 'lowpass' | 'highpass' | 'none';
  description: string;
  low_cutoff?: number;
  high_cutoff?: number;
  cutoff?: number;
};

type PresetMap = Record<string, FilterPreset>;

/** Fallback presets (used if backend fetch fails) */
const FALLBACK_PRESETS: PresetMap = {
  
  telephone: {
    filter_type: 'bandpass',
    low_cutoff: 300,
    high_cutoff: 3400,
    description: 'Telephone bandwidth (300–3400 Hz) — removes very low rumble and high frequencies beyond voice range.',
  },
  speech: {
    filter_type: 'bandpass',
    low_cutoff: 80,
    high_cutoff: 8000,
    description: 'Typical speech range (80–8000 Hz) — keeps full speech clarity while removing unnecessary extremes.',
  },
  voice_fundamental: {
    filter_type: 'bandpass',
    low_cutoff: 85,
    high_cutoff: 255,
    description: 'Fundamental frequency of human voice (approx. 85–180 Hz male, 165–255 Hz female).',
  },
  noise_reduction: {
    filter_type: 'lowpass',
    cutoff: 4000,
    description: 'Reduce high-frequency noise by keeping content below 4 kHz.',
  },
  no_filter: {
    filter_type: 'highpass',
    cutoff: 100,
    description: 'Remove low-frequency rumble and mic handling noise (<100 Hz).',
  },


};

const BASE_URL ="https://06cc7a0499c3.ngrok-free.app";

const DetectEmotion = () => {
  const [emotionResult, setEmotionResult] = useState<EmotionData | null>(null);
  const [genderResult, setGenderResult] = useState<GenderData | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisType, setAnalysisType] = useState<'both' | 'emotion' | 'gender'>('both');

  // DSP preset state
  const [filterPresets, setFilterPresets] = useState<PresetMap>({});
  const [selectedPreset, setSelectedPreset] = useState<string>(''); // choose after presets load
  const [presetsLoading, setPresetsLoading] = useState<boolean>(true);
  const [presetsError, setPresetsError] = useState<string | null>(null);
  
  // DSP method state (hardcoded vs library)
  const [useHardcodedDSP, setUseHardcodedDSP] = useState<boolean>(true);
  
  // Detection method state (ML vs DSP)
  const [detectionMethod, setDetectionMethod] = useState<'ml' | 'dsp'>('ml');

  // --- Load presets from backend with safe defaults ---
  useEffect(() => {
    let cancelled = false;

    const loadPresets = async () => {
      setPresetsLoading(true);
      setPresetsError(null);

      try {
        const res = await fetch(`${BASE_URL}/filter-presets`);
        const data = await res.json();

        // Defensive: make sure structure is as expected
        const presets: PresetMap =
          data && data.presets && typeof data.presets === 'object'
            ? data.presets
            : FALLBACK_PRESETS;

        if (!cancelled) {
          setFilterPresets(presets);

          // Prefer 'telephone' if present, else first key
          const defaultKey =
            'telephone' in presets
              ? 'telephone'
              : Object.keys(presets)[0] || 'no_filter';

          setSelectedPreset(defaultKey);
        }
      } catch (e) {
        // Fallback silently to built-ins
        console.log(e);
        if (!cancelled) {
          setFilterPresets(FALLBACK_PRESETS);
          setSelectedPreset('telephone');
          setPresetsError('Using fallback presets (backend presets unavailable).');
        }
      } finally {
        if (!cancelled) setPresetsLoading(false);
      }
    };

    loadPresets();
    return () => {
      cancelled = true;
    };
  }, []);

  // Build query params from current preset
  const buildQueryFromPreset = (preset?: FilterPreset) => {
    const qp = new URLSearchParams();

    // Always be explicit (backend defaults to True but we make intent clear)
    qp.set('apply_filtering', 'true');

    if (!preset) return qp;

    qp.set('filter_type', preset.filter_type);

    // Only include present parameters; backend validates combinations
    if (typeof preset.low_cutoff === 'number') {
      qp.set('low_cutoff', String(preset.low_cutoff));
    }
    if (typeof preset.high_cutoff === 'number') {
      qp.set('high_cutoff', String(preset.high_cutoff));
    }
    if (typeof preset.cutoff === 'number') {
      qp.set('cutoff', String(preset.cutoff));
    }

    // Add DSP method parameter
    qp.set('use_hardcoded_dsp', String(useHardcodedDSP));
    
    // Add detection method parameter
    qp.set('detection_method', detectionMethod);

    return qp;
  };

  const handleAudioUpload = async (audioBlob: Blob) => {
    setIsAnalyzing(true);
    setEmotionResult(null);
    setGenderResult(null);

    try {
      const promises: Array<
        Promise<
          | { type: 'emotion'; data: EmotionData }
          | { type: 'gender'; data: GenderData }
        >
      > = [];

      const preset = filterPresets[selectedPreset];
      const queryParams = buildQueryFromPreset(preset).toString();

      if (analysisType === 'emotion' || analysisType === 'both') {
        const emotionFormData = new FormData();
        emotionFormData.append('audio', audioBlob, 'recording.wav');

        promises.push(
          fetch(`${BASE_URL}/predict?${queryParams}`, {
            method: 'POST',
            body: emotionFormData,
          })
            .then(async (res) => {
              if (!res.ok) {
                throw new Error(`HTTP error! status: ${res.status}`);
              }
              const text = await res.text();
              try {
                return JSON.parse(text);
              } catch (e) {
                console.error('Failed to parse JSON response:', text, e);
                throw new Error('Invalid JSON response from server');
              }
            })
            .then((data) => ({ type: 'emotion', data }))
        );
      }

      if (analysisType === 'gender' || analysisType === 'both') {
        const genderFormData = new FormData();
        genderFormData.append('audio', audioBlob, 'recording.wav');

        promises.push(
          fetch(`${BASE_URL}/classify-gender?${queryParams}`, {
            method: 'POST',
            body: genderFormData,
          })
            .then(async (res) => {
              if (!res.ok) {
                throw new Error(`HTTP error! status: ${res.status}`);
              }
              const text = await res.text();
              try {
                return JSON.parse(text);
              } catch (e) {
                console.error('Failed to parse JSON response:', text, e);
                throw new Error('Invalid JSON response from server');
              }
            })
            .then((data) => ({ type: 'gender', data }))
        );
      }

      const results = await Promise.all(promises);
      results.forEach((result) => {
        if (result.type === 'emotion') {
          setEmotionResult(result.data);
        } else if (result.type === 'gender') {
          setGenderResult(result.data);
        }
      });
    } catch (error) {
      console.error('Error uploading audio:', error);

      // Fallback UI-friendly errors (kept from your original)
      if (analysisType === 'emotion' || analysisType === 'both') {
        setEmotionResult({
          emotion: 'Error',
          confidence: 0,
          topEmotions: [
            { emotion: 'Error', score: 1.0 },
            { emotion: 'Please start backend', score: 0.0 },
            { emotion: 'See README', score: 0.0 },
          ],
        });
      }
      if (analysisType === 'gender' || analysisType === 'both') {
        setGenderResult({
          gender: 'Error',
          confidence: 0,
          method: 'error',
          scores: { male_score: 0, female_score: 0 },
          feature_analysis: {
            f0_hz: 0,
            spectral_centroid_hz: 0,
            f1_hz: 0,
            f2_hz: 0
          },
          all_features: {}
        });
      }
    } finally {
      setIsAnalyzing(false);
    }
  };

  const currentPreset: FilterPreset | undefined = selectedPreset
    ? filterPresets[selectedPreset]
    : undefined;

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
              { key: 'gender', label: 'Gender Only', icon: <Users className="w-4 h-4" /> },
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

        {/* DSP Filter Preset Selector */}
        {/* DSP Filter Preset Selector */}
<div className="bg-white/5 backdrop-blur-sm rounded-3xl p-6 border border-white/10 mb-8">
  <div className="flex items-center justify-between">
    <h2 className="text-xl font-bold text-white mb-4">DSP Filter Preset</h2>
    {presetsLoading && (
      <span className="text-sm text-blue-200">Loading presets…</span>
    )}
  </div>

  <div className="flex items-center gap-4">
    <select
      disabled={presetsLoading || !Object.keys(filterPresets).length}
      value={selectedPreset}
      onChange={(e) => setSelectedPreset(e.target.value)}
      className={`px-4 py-2 rounded-xl bg-white/10 text-blue-200 ${
        presetsLoading ? 'opacity-60 cursor-not-allowed' : ''
      }`}
    >
      {/* Hardcode ONLY names (no descriptions) */}
      <option value="telephone">Telephone</option>
      <option value="speech">Speech</option>
      <option value="voice_fundamental">Voice Fundamental</option>
      <option value="noise_reduction">Noise Reduction</option>
      <option value="no_filter">No Filter</option>
    </select>
  </div>

  {presetsError && (
    <p className="mt-2 text-xs text-yellow-300">{presetsError}</p>
  )}
</div>


        {/* Detection Method Selector */}
        <div className="bg-white/5 backdrop-blur-sm rounded-3xl p-6 border border-white/10 mb-8">
          <h2 className="text-xl font-bold text-white mb-4">Emotion Detection Method</h2>
          <div className="flex gap-4">
            <button
              onClick={() => setDetectionMethod('ml')}
              className={`flex items-center gap-2 px-4 py-2 rounded-xl transition-all duration-200 ${
                detectionMethod === 'ml'
                  ? 'bg-indigo-500 text-white shadow-lg'
                  : 'bg-white/10 text-blue-200 hover:bg-white/20'
              }`}
            >
              <span className="w-2 h-2 rounded-full bg-current"></span>
              🤖 ML Model
            </button>
          </div>
          <p className="text-sm text-blue-200 mt-2">
            Using emotion2vec_plus_large machine learning model for emotion detection
          </p>
        </div>

        {/* DSP Preprocessing Method Selector - show for both ML and DSP detection methods */}
        <div className="bg-white/5 backdrop-blur-sm rounded-3xl p-6 border border-white/10 mb-8">
          <h2 className="text-xl font-bold text-white mb-4">DSP Preprocessing Method</h2>
          <div className="flex gap-4">
            <button
              onClick={() => setUseHardcodedDSP(true)}
              className={`flex items-center gap-2 px-4 py-2 rounded-xl transition-all duration-200 ${
                useHardcodedDSP
                  ? 'bg-green-500 text-white shadow-lg'
                  : 'bg-white/10 text-blue-200 hover:bg-white/20'
              }`}
            >
              <span className="w-2 h-2 rounded-full bg-current"></span>
              Hardcoded DSP
            </button>
            <button
              onClick={() => setUseHardcodedDSP(false)}
              className={`flex items-center gap-2 px-4 py-2 rounded-xl transition-all duration-200 ${
                !useHardcodedDSP
                  ? 'bg-purple-500 text-white shadow-lg'
                  : 'bg-white/10 text-blue-200 hover:bg-white/20'
              }`}
            >
              <span className="w-2 h-2 rounded-full bg-current"></span>
              Library Functions
            </button>
          </div>
          <p className="text-sm text-blue-200 mt-2">
            {useHardcodedDSP 
              ? "Using custom hardcoded DSP implementations for preprocessing (FFT, filters, etc.)"
              : "Using SciPy and NumPy library functions for DSP preprocessing"
            }
          </p>
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
