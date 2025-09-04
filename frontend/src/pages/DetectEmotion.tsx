import { useState, useEffect } from 'react';
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
    description: 'Telephone bandwidth (300–3400 Hz)',
  },
  speech: {
    filter_type: 'bandpass',
    low_cutoff: 85,
    high_cutoff: 8000,
    description: 'Human speech range (85–8000 Hz)',
  },
  voice_fundamental: {
    filter_type: 'bandpass',
    low_cutoff: 80,
    high_cutoff: 1000,
    description: 'Voice fundamental (80–1000 Hz)',
  },
  noise_reduction: {
    filter_type: 'highpass',
    cutoff: 100,
    description: 'Remove low-frequency noise (>100 Hz)',
  },
  no_filter: {
    filter_type: 'none',
    description: 'No filtering',
  },
};

const BASE_URL = 'https://516bd481df3d.ngrok-free.app';

const DetectEmotion = () => {
  const [emotionResult, setEmotionResult] = useState<EmotionData | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  // DSP preset state
  const [filterPresets, setFilterPresets] = useState<PresetMap>({});
  const [selectedPreset, setSelectedPreset] = useState<string>(''); // choose after presets load
  const [presetsLoading, setPresetsLoading] = useState<boolean>(true);
  const [presetsError, setPresetsError] = useState<string | null>(null);

  // --- Load presets from backend with safe defaults ---
  useEffect(() => {
    let cancelled = false;

    const loadPresets = async () => {
      setPresetsLoading(true);
      setPresetsError(null);

      try {
        const res = await fetch(`${BASE_URL}/filter-presets`);
        const data = await res.json();

        const presets: PresetMap =
          data && data.presets && typeof data.presets === 'object'
            ? data.presets
            : FALLBACK_PRESETS;

        if (!cancelled) {
          setFilterPresets(presets);

          const defaultKey =
            'telephone' in presets
              ? 'telephone'
              : Object.keys(presets)[0] || 'no_filter';

          setSelectedPreset(defaultKey);
        }
      } catch (e) {
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

  const buildQueryFromPreset = (preset?: FilterPreset) => {
    const qp = new URLSearchParams();
    qp.set('apply_filtering', 'true');

    if (!preset) return qp;

    qp.set('filter_type', preset.filter_type);
    if (typeof preset.low_cutoff === 'number') qp.set('low_cutoff', String(preset.low_cutoff));
    if (typeof preset.high_cutoff === 'number') qp.set('high_cutoff', String(preset.high_cutoff));
    if (typeof preset.cutoff === 'number') qp.set('cutoff', String(preset.cutoff));

    return qp;
  };

  const handleAudioUpload = async (audioBlob: Blob) => {
    setIsAnalyzing(true);
    setEmotionResult(null);

    try {
      const preset = filterPresets[selectedPreset];
      const queryParams = buildQueryFromPreset(preset).toString();

      const emotionFormData = new FormData();
      emotionFormData.append('audio', audioBlob, 'recording.wav');

      const res = await fetch(`${BASE_URL}/predict?${queryParams}`, {
        method: 'POST',
        body: emotionFormData,
      });
      const data = await res.json();
      setEmotionResult(data);
    } catch (error) {
      console.error('Error uploading audio:', error);
      setEmotionResult({
        emotion: 'Error',
        confidence: 0,
        topEmotions: [
          { emotion: 'Error', score: 1.0 },
          { emotion: 'Please start backend', score: 0.0 },
          { emotion: 'See README', score: 0.0 },
        ],
      });
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
              {Object.entries(filterPresets).map(([key, preset]) => (
                <option key={key} value={key}>
                  {key.replace(/_/g, ' ')} — {preset.description}
                </option>
              ))}
            </select>

            {currentPreset && (
              <div className="text-blue-200 text-sm">
                <span className="block">
                  <b>Type:</b> {currentPreset.filter_type}
                </span>
                {currentPreset.filter_type === 'bandpass' && (
                  <span className="block">
                    <b>Band:</b> {currentPreset.low_cutoff}–{currentPreset.high_cutoff} Hz
                  </span>
                )}
                {currentPreset.filter_type !== 'bandpass' && currentPreset.cutoff !== undefined && (
                  <span className="block">
                    <b>Cutoff:</b> {currentPreset.cutoff} Hz
                  </span>
                )}
              </div>
            )}
          </div>

          {presetsError && (
            <p className="mt-2 text-xs text-yellow-300">{presetsError}</p>
          )}
        </div>

        <div className="bg-white/5 backdrop-blur-sm rounded-3xl p-8 border border-white/10">
          <h2 className="text-2xl font-bold text-white mb-6">Record Your Voice</h2>
          <p className="text-blue-100 mb-8">
            Click the record button and speak naturally. We'll analyze your speech using advanced DSP techniques.
          </p>
          <AudioRecorder onAudioReady={handleAudioUpload} />
        </div>

        <div className="bg-white/5 backdrop-blur-sm rounded-3xl p-8 border border-white/10 mt-8">
          <h2 className="text-2xl font-bold text-white mb-6">Analysis Results</h2>

          {isAnalyzing && (
            <div className="text-center py-12">
              <div className="inline-flex items-center gap-3 text-blue-200">
                <div className="animate-spin rounded-full h-6 w-6 border-2 border-blue-400 border-t-transparent"></div>
                Analyzing speech...
              </div>
            </div>
          )}

          {!isAnalyzing && !emotionResult && (
            <div className="text-center py-12 text-blue-200">
              Record audio to see analysis results here
            </div>
          )}

          {!isAnalyzing && emotionResult && (
            <div>
              <div className="flex items-center gap-2 mb-4">
                <Brain className="w-5 h-5 text-blue-400" />
                <h3 className="text-lg font-semibold text-white">Emotion Detection</h3>
              </div>
              <EmotionResult data={emotionResult} />
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default DetectEmotion;
