import React, { useEffect, useMemo, useState } from 'react';
import { fetchAudioDevices, startMeeting } from '../services/api';
import type { AudioDeviceInfo, AudioSource } from '../types';
import { useNavigate } from 'react-router-dom';

export default function Setup() {
  const navigate = useNavigate();
  const [topic, setTopic] = useState('');
  const [participants, setParticipants] = useState(4);
  const [source, setSource] = useState<AudioSource>('mic');
  const [devices, setDevices] = useState<AudioDeviceInfo[]>([]);
  const [deviceIndex, setDeviceIndex] = useState<number | undefined>(undefined);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    (async () => {
      const list = await fetchAudioDevices();
      setDevices(list.filter(d => (d?.max_input_channels ?? 0) > 0));
    })();
  }, []);

  const canStart = topic.trim().length > 0 && !busy;

  const handleStart = async () => {
    if (!canStart) return;
    setBusy(true);
    try {
      const ok = await startMeeting({
        topic: topic.trim(),
        participants: Math.max(0, Number(participants) || 0),
        source,
        device_index: deviceIndex,
      });
      if (!ok) throw new Error('Failed to start meeting');
      navigate('/dashboard');
    } catch (e) {
      const m = e instanceof Error ? e.message : 'Unknown error';
      alert(m);
    } finally {
      setBusy(false);
    }
  };

  const loopbackHint = useMemo(
    () => (
      source === 'system' ? (
        <p className="text-xs text-gray-500 mt-2">
          Tip: Install a loopback input (e.g., BlackHole 2ch) and create a Multi-Output Device on macOS.
        </p>
      ) : null
    ),
    [source],
  );

  return (
    <div className="min-h-screen bg-[#fafbfc] flex items-center justify-center px-4">
      <div className="w-full max-w-xl bg-white rounded-xl border border-[#e5e7eb] p-6 shadow-sm">
        <h1 className="text-xl font-semibold mb-4">Pre‑meeting Setup</h1>

        <div className="space-y-4">
          <div>
            <label className="block text-sm text-[#374151] mb-1">Meeting Topic</label>
            <input
              className="w-full border rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
              placeholder="e.g., Sprint Planning"
              value={topic}
              onChange={e => setTopic(e.target.value)}
            />
          </div>

          <div>
            <label className="block text-sm text-[#374151] mb-1">Participant Count</label>
            <input
              type="number"
              min={1}
              className="w-32 border rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
              value={participants}
              onChange={e => setParticipants(Number(e.target.value))}
            />
          </div>

          <div>
            <label className="block text-sm text-[#374151] mb-1">Audio Source</label>
            <div className="flex items-center gap-4">
              <label className="flex items-center gap-2 text-sm">
                <input
                  type="radio"
                  name="source"
                  checked={source === 'mic'}
                  onChange={() => setSource('mic')}
                />
                <span>Microphone</span>
              </label>
              <label className="flex items-center gap-2 text-sm">
                <input
                  type="radio"
                  name="source"
                  checked={source === 'system'}
                  onChange={() => setSource('system')}
                />
                <span>System (loopback)</span>
              </label>
            </div>
            {loopbackHint}
          </div>

          <div>
            <label className="block text-sm text-[#374151] mb-1">Input Device (optional)</label>
            <select
              className="w-full border rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
              value={deviceIndex ?? ''}
              onChange={(e) => {
                const v = e.target.value;
                setDeviceIndex(v === '' ? undefined : Number(v));
              }}
            >
              <option value="">Auto select</option>
              {devices.map(d => (
                <option key={d.index} value={d.index}>{`${d.name} — ${d.hostapi}`}</option>
              ))}
            </select>
          </div>

          <div className="pt-2">
            <button
              onClick={handleStart}
              disabled={!canStart}
              className={`px-4 py-2 rounded-md text-white ${canStart ? 'bg-blue-600 hover:bg-blue-700' : 'bg-gray-300'}`}
            >
              {busy ? 'Starting…' : 'Start Meeting'}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

