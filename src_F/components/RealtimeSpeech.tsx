import { useState, useEffect } from 'react';
import { AudioVisualizer } from './AudioVisualizer';
import { StatusIndicator } from './StatusIndicator';

interface SpeechEntry {
  id: number;
  speaker: string;
  text: string;
  timestamp: string;
}

interface RealtimeSpeechProps {
  isActive: boolean;
}

export function RealtimeSpeech({ isActive }: RealtimeSpeechProps) {
  const [timer, setTimer] = useState('00:00:00');
  const [speechEntries, setSpeechEntries] = useState<SpeechEntry[]>([]);
  const [currentSpeaker, setCurrentSpeaker] = useState('김팀장');

  useEffect(() => {
    if (!isActive) {
      setTimer('00:00:00');
      return;
    }

    const startTime = Date.now();
    const interval = setInterval(() => {
      const elapsed = Date.now() - startTime;
      const seconds = Math.floor(elapsed / 1000);
      const minutes = Math.floor(seconds / 60);
      const hours = Math.floor(minutes / 60);
      
      setTimer(
        `${hours.toString().padStart(2, '0')}:${(minutes % 60).toString().padStart(2, '0')}:${(seconds % 60).toString().padStart(2, '0')}`
      );
    }, 1000);

    return () => clearInterval(interval);
  }, [isActive]);

  useEffect(() => {
    if (!isActive) {
      setSpeechEntries([]);
      return;
    }

    const speakers = ['김팀장', '이대리', '박과장', '최사원'];
    const sampleTexts = [
      '프로젝트 일정에 대해 논의하겠습니다.',
      '이번 분기 목표 달성을 위한 전략이 필요합니다.',
      '고객 피드백을 반영한 개선사항을 검토해보겠습니다.',
      '마케팅 캠페인 효과를 분석한 결과입니다.',
      '다음 주 마일스톤 달성을 위한 계획을 세워야 합니다.'
    ];

    const interval = setInterval(() => {
      const speaker = speakers[Math.floor(Math.random() * speakers.length)];
      const text = sampleTexts[Math.floor(Math.random() * sampleTexts.length)];
      const now = new Date();
      const timestamp = `${now.getHours().toString().padStart(2, '0')}:${now.getMinutes().toString().padStart(2, '0')}`;
      
      setCurrentSpeaker(speaker);
      setSpeechEntries(prev => [
        ...prev,
        {
          id: Date.now(),
          speaker,
          text,
          timestamp
        }
      ].slice(-10)); // Keep only last 10 entries
    }, 3000);

    return () => clearInterval(interval);
  }, [isActive]);

  return (
    <div className="bg-white border border-[#e5e7eb] rounded-xl p-6 shadow-sm hover:shadow-md transition-shadow h-full flex flex-col">
      <div className="flex items-center justify-between mb-6">
        <h2>실시간 발언</h2>
        <StatusIndicator isActive={isActive} currentSpeaker={isActive ? currentSpeaker : undefined} />
      </div>
      
      <AudioVisualizer isActive={isActive} />
      
      <div className="flex-1 bg-white p-3 my-4 rounded-lg overflow-y-auto">
        <div className="space-y-3 max-h-72 overflow-y-auto scrollbar-thin scrollbar-track-transparent scrollbar-thumb-[#d1d1d1]">
          {speechEntries.map((entry) => (
            <div
              key={entry.id}
              className="p-3 rounded-lg hover:bg-[#f0f9ff] transition-colors"
            >
              <div className="flex items-center justify-between mb-1">
                <span className="text-sm text-[#374151]">{entry.speaker}</span>
                <span className="text-xs text-[#6b7280]">{entry.timestamp}</span>
              </div>
              <p className="text-sm text-[#1a1a1a]">{entry.text}</p>
            </div>
          ))}
          {speechEntries.length === 0 && (
            <div className="text-center text-[#6b7280] py-8">
              {isActive ? '발언을 분석하고 있습니다...' : '회의를 시작하면 실시간 발언이 표시됩니다.'}
            </div>
          )}
        </div>
      </div>
      
      <div className="border-t border-[#e5e7eb] pt-3 text-right">
        <span className="text-[#6b7280] font-mono">{timer}</span>
      </div>
    </div>
  );
}