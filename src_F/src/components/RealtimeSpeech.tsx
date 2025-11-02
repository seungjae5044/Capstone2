import React, { useCallback, useEffect, useMemo, useState ,useRef} from 'react';
import { AudioVisualizer } from './AudioVisualizer';
import type { TranscriptionEntry } from '../types';
import * as ScrollAreaPrimitive from "@radix-ui/react-scroll-area";

const SPEAKER_COLORS = ['#FF4444', '#44FF44', '#4444FF', '#FFFF44', '#FF44FF', '#44FFFF', '#FF8844', '#FF009D', '#8844FF', '#FFAA44'];

interface RealtimeSpeechProps {
  isActive: boolean;
  transcriptions: TranscriptionEntry[];
  speakerColors: Record<string, string>;
}

export function RealtimeSpeech({ isActive, transcriptions, speakerColors }: RealtimeSpeechProps) {
  const [timer, setTimer] = useState(0);
  const scrollRef = useRef<HTMLDivElement | null>(null);
  const [atBottom, setAtBottom] = useState(true);

  useEffect(() => {
    if (!isActive) {
      setTimer(0);
      return;
    }

    const interval = setInterval(() => setTimer(t => t + 1), 1000);
    return () => clearInterval(interval);
  }, [isActive]);

  const formatTime = (seconds: number) => {
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    const s = seconds % 60;
    return `${h.toString().padStart(2, '0')}:${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
  };

  const colorMap = useMemo(() => speakerColors, [speakerColors]);

  const meetingStartTime = useMemo(() => {
  if (transcriptions.length === 0) return null;
  return new Date(transcriptions[0].timestamp).getTime();
}, [transcriptions]);

const formatTimestamp = (timestamp: string) => {
  if (!meetingStartTime) return '00:00:00';
  
  const currentTime = new Date(timestamp).getTime();
  const elapsedSeconds = Math.floor((currentTime - meetingStartTime) / 1000);
  
  const h = Math.floor(elapsedSeconds / 3600);
  const m = Math.floor((elapsedSeconds % 3600) / 60);
  const s = elapsedSeconds % 60;
  
  return `${h.toString().padStart(2, '0')}:${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
};

  // Auto-scroll to bottom when new items arrive and user is already at bottom
  useEffect(() => {
    if (!scrollRef.current) return;
    if (atBottom) {
      const el = scrollRef.current;
      el.scrollTo({ top: el.scrollHeight, behavior: 'smooth' });
    }
  }, [transcriptions, atBottom]);

  const onScroll = () => {
    const el = scrollRef.current;
    if (!el) return;
    const isAtBottom = Math.abs(el.scrollHeight - el.scrollTop - el.clientHeight) < 4;
    setAtBottom(isAtBottom);
  };

  const scrollToTop = () => {
    scrollRef.current?.scrollTo({ top: 0, behavior: 'smooth' });
  };
  const scrollToBottom = () => {
    const el = scrollRef.current;
    if (!el) return;
    el.scrollTo({ top: el.scrollHeight, behavior: 'smooth' });
  };

  return (
    <div className="bg-white border border-[#e5e7eb] rounded-xl p-6 h-full min-h-0 flex flex-col relative max-h-[600px]">
      <div className="flex items-center justify-between mb-4">
        <h2>실시간 발언</h2>
        <div className="flex items-center gap-2">
          <div className={`w-2 h-2 rounded-full ${isActive ? 'bg-green-500 animate-pulse' : 'bg-gray-300'}`} />
          <span className="text-sm text-[#6b7280]">{isActive ? '녹화 중' : '대기 중'}</span>
        </div>
      </div>

      <AudioVisualizer isActive={isActive} />

      <div className="flex-1 flex flex-col min-h-0">
      <ScrollAreaPrimitive.Root style={{ height: 400 }}>
      <ScrollAreaPrimitive.Viewport
          ref={scrollRef}
          onScroll={onScroll}
          className="h-full w-full pr-2 space-y-2"
        >
          {transcriptions.map((entry) => (
            <div
              key={entry.id}
              className="p-3 rounded-lg hover:bg-[#f0f9ff] transition-colors"
              style={{
                borderLeft: `3px solid ${colorMap[entry.speakerId] ?? SPEAKER_COLORS[0]}`,
              }}
            >
              <div className="flex items-center justify-between mb-1">
                <span className="text-sm">{entry.speakerName}</span>
                <span className="text-xs text-[#6b7280]">{formatTimestamp(entry.timestamp)}</span>
              </div>
              <p className="text-sm text-[#1a1a1a] mb-2">{entry.text}</p>
              {(entry.topicScore ?? entry.noveltyScore ?? entry.similarity) !== undefined && (
                <div className="flex gap-4 text-xs text-[#6b7280] flex-wrap">
                  <span>
                    주제: {entry.topicScore !== undefined ? entry.topicScore.toFixed(1) : '측정 중'}
                  </span>
                  <span>
                    신규성: {entry.noveltyScore !== undefined ? entry.noveltyScore.toFixed(1) : '측정 중'}
                  </span>
                  {entry.similarity !== undefined && (
                    <span>신뢰도: {(entry.similarity * 100).toFixed(0)}%</span>
                  )}
                  {entry.duration !== undefined && (
                    <span>길이: {entry.duration.toFixed(1)}초</span>
                  )}
                </div>
              )}
            </div>
          ))}
          {transcriptions.length === 0 && (
            <div className="text-center text-[#6b7280] py-12">
              {isActive ? '발언을 분석하고 있습니다...' : '회의를 시작하면 실시간 발언이 표시됩니다.'}
            </div>
          )}
        </ScrollAreaPrimitive.Viewport>
        <ScrollAreaPrimitive.Scrollbar orientation="vertical" />
        <ScrollAreaPrimitive.Corner />
      </ScrollAreaPrimitive.Root>
      </div>

      <div className="border-t border-[#e5e7eb] pt-3 text-right mt-4">
        <span className="text-[#6b7280] font-mono">{formatTime(timer)}</span>
      </div>
    </div>
  );
}
