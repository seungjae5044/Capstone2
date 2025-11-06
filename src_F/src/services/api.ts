import type { AudioDeviceInfo, StartRequest } from '../types';

export async function fetchAudioDevices(): Promise<AudioDeviceInfo[]> {
  try {
    const r = await fetch('/audio/devices');
    if (!r.ok) return [];
    const j = await r.json();
    return Array.isArray(j?.devices) ? j.devices : [];
  } catch {
    return [];
  }
}

export async function startMeeting(body: StartRequest): Promise<boolean> {
  const r = await fetch('/start', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  return r.ok;
}

