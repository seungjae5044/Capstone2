import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { Header } from './components/Header';
import { RealtimeSpeech } from './components/RealtimeSpeech';
import { Statistics } from './components/Statistics';
import { SpeakerAnalysisTable } from './components/SpeakerAnalysisTable';
import { TimelineVisualization } from './components/TimelineVisualization';
import { WebSocketClient } from './services/websocket';
import type {
  SpeakerStats,
  TimelineEntry,
  TimelineSegmentMessage,
  TranscriptionEntry,
  WebSocketMessage,
} from './types';

export default function App() {
  const [speakerColors, setSpeakerColors] = useState<Record<string, string>>({});
  const [isActive, setIsActive] = useState(false);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [transcriptions, setTranscriptions] = useState<TranscriptionEntry[]>([]);
  const [speakers, setSpeakers] = useState<SpeakerStats[]>([]);
  const [avgTopic, setAvgTopic] = useState(0);
  const [avgNovelty, setAvgNovelty] = useState(0);
  const [timeline, setTimeline] = useState<TimelineEntry[]>([]);
  const [ws] = useState(() => new WebSocketClient());

  const colorPalette = useMemo(
    () => [
      '#2563eb',
      '#16a34a',
      '#db2777',
      '#f59e0b',
      '#8b5cf6',
      '#06b6d4',
      '#f97316',
      '#84cc16',
      '#ec4899',
      '#14b8a6',
      '#f43f5e',
      '#0ea5e9',
    ],
    [],
  );

  const ensureSpeakerColor = useCallback(
    (speakerId: string) => {
      if (!speakerId) {
        return;
      }
      setSpeakerColors(prev => {
        if (prev[speakerId]) {
          return prev;
        }
        const nextIndex = Object.keys(prev).length % colorPalette.length;
        return {
          ...prev,
          [speakerId]: colorPalette[nextIndex],
        };
      });
    },
    [colorPalette],
  );

  useEffect(() => {
    ws.connect((message: WebSocketMessage) => {
      switch (message.type) {
        case 'transcription': {
          ensureSpeakerColor(message.speaker_id);
          const entry: TranscriptionEntry = {
            id: `${message.timestamp}-${Math.random().toString(36).slice(2, 8)}`,
            speakerId: message.speaker_id,
            speakerName: message.speaker_name ?? message.speaker_id,
            text: message.text,
            timestamp: message.timestamp,
            topicScore: message.topic_score,
            noveltyScore: message.novelty_score,
            similarity: message.similarity,
            duration: message.duration,
          };
          setTranscriptions(prev => [...prev, entry].slice(-20));
          break;
        }
        case 'evaluation': {
          ensureSpeakerColor(message.speaker_id);
          setTranscriptions(prev => {
            const next = [...prev];
            for (let i = next.length - 1; i >= 0; i -= 1) {
              const candidate = next[i];
              const isSameSpeaker = candidate.speakerId === message.speaker_id;
              const isSameText = !message.text || candidate.text === message.text;
              if (isSameSpeaker && isSameText) {
                next[i] = {
                  ...candidate,
                  topicScore: message.topic_relevance,
                  noveltyScore: message.novelty,
                  speakerName: message.speaker_name ?? candidate.speakerName,
                };
                break;
              }
            }
            return next;
          })
        .catch(() => undefined);
          break;
        }
        case 'stats': {
          message.speakers.forEach(item => ensureSpeakerColor(item.speaker_id));
          setSpeakers(message.speakers);
          setAvgTopic(message.avg_topic);
          setAvgNovelty(message.avg_novelty);
          break;
        }
        case 'session_status': {
          setIsActive(message.is_active);
          setSessionId(message.session_id ?? null);
          if (!message.is_active) {
            setTranscriptions([]);
            setTimeline([]);
          }
          break;
        }
        case 'timeline_segment': {
          const { segment } = message as TimelineSegmentMessage;
          ensureSpeakerColor(segment.speaker_id);
          setTimeline(prev => {
            const next = [...prev];
            const candidate: TimelineEntry = {
              speakerId: segment.speaker_id,
              speakerName: segment.speaker_name,
              startTime: segment.start_time,
              endTime: segment.end_time,
              duration: segment.duration,
            };
            const last = next[next.length - 1];
            if (
              last &&
              last.speakerId === candidate.speakerId &&
              last.startTime === candidate.startTime
            ) {
              next[next.length - 1] = candidate;
            } else {
              next.push(candidate);
            }
            return next;
          });
          break;
        }
        default:
          break;
      }
    });

    return () => ws.disconnect();
  }, [ws]);

  const handleStart = async (topic: string, participants: number) => {
    try {
      setTranscriptions([]);
      setSpeakers([]);
      const response = await fetch('/api/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          topic,
          speaker_id: 'Speaker 1',
          participants,
        }),
      });

      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.detail || '회의 시작에 실패했습니다.');
      }
      setIsActive(true);
      setSessionId(payload.session_id ?? null);
      await Promise.allSettled([
        fetch('/api/timeline')
          .then(res => (res.ok ? res.json() : null))
          .then(data => {
            if (data?.segments) {
              const entries: TimelineEntry[] = data.segments.map((segment: any) => ({
                speakerId: segment.speaker_id,
                speakerName: segment.speaker_name,
                startTime: segment.start_time,
                endTime: segment.end_time,
                duration: segment.duration,
              }));
              setTimeline(entries);
              entries.forEach(entry => ensureSpeakerColor(entry.speakerId));
            }
          }),
        fetch('/api/speakers')
          .then(res => (res.ok ? res.json() : null))
          .then(data => {
            if (data?.speakers) {
              data.speakers.forEach((speaker: any) => ensureSpeakerColor(speaker.speaker_id));
            }
          }),
      ]);
    } catch (error) {
      const message = error instanceof Error ? error.message : '알 수 없는 오류가 발생했습니다.';
      alert(message);
      setIsActive(false);
    }
  };

  const handleStop = async () => {
    if (!sessionId) {
      return;
    }
    try {
      const response = await fetch(`/api/stop/${sessionId}`, { method: 'POST' });
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.detail || '회의 중지에 실패했습니다.');
      }
      setIsActive(false);
      setSessionId(null);
      fetch('/api/speakers')
        .then(res => (res.ok ? res.json() : null))
        .then(data => {
          if (data?.stats) {
            const speakerList: SpeakerStats[] = Object.entries(data.stats).map(([id, stats]: [string, any]) => ({
              speaker_id: id,
              name: data.speakers?.find((item: any) => item.speaker_id === id)?.name ?? id,
              count: stats.total_statements ?? 0,
              duration: stats.total_duration ?? 0,
              topic_avg: stats.avg_topic_relevance ?? 0,
              novelty_avg: stats.avg_novelty ?? 0,
              participation: 0,
            }));
            const totalDuration = speakerList.reduce((acc, item) => acc + item.duration, 0) || 0;
            const normalized = speakerList.map(item => ({
              ...item,
              participation: totalDuration ? (item.duration / totalDuration) * 100 : 0,
            }));
            setSpeakers(normalized);
            normalized.forEach(item => ensureSpeakerColor(item.speaker_id));
            if (data.stats_overall) {
              setAvgTopic(data.stats_overall.avg_topic_relevance ?? 0);
              setAvgNovelty(data.stats_overall.avg_novelty ?? 0);
            }
          }
        })
        .catch(() => undefined);
    } catch (error) {
      const message = error instanceof Error ? error.message : '알 수 없는 오류가 발생했습니다.';
      alert(message);
    }
  };

  const handleReport = () => {
    window.open('/api/report', '_blank');
  };

  useEffect(() => {
    if (!sessionId) {
      return;
    }
    (async () => {
      try {
        const response = await fetch('/api/timeline');
        if (!response.ok) {
          return;
        }
        const data = await response.json();
        if (data?.segments) {
          const entries: TimelineEntry[] = data.segments.map((segment: any) => ({
            speakerId: segment.speaker_id,
            speakerName: segment.speaker_name,
            startTime: segment.start_time,
            endTime: segment.end_time,
            duration: segment.duration,
          }));
          setTimeline(entries);
          entries.forEach(entry => ensureSpeakerColor(entry.speakerId));
        }
      } catch (error) {
        console.error('Failed to fetch timeline', error);
      }
    })();
  }, [sessionId, ensureSpeakerColor]);

  return (
    <div className="h-screen bg-[#fafbfc] flex flex-col">
      <Header
        isActive={isActive}
        onStart={handleStart}
        onStop={handleStop}
        onReport={handleReport}
      />

      {/* main 태그를 flex-1로 만들어 Header를 제외한 나머지 공간을 모두 차지하게 합니다. */}
      <main className="max-w-7xl w-full mx-auto px-6 py-6 flex-1 flex flex-col min-h-0">
        {/* RealtimeSpeech와 Statistics를 담는 grid가 남은 공간을 채우도록 flex-1을 추가합니다. */}
        <div className="grid grid-cols-12 gap-6 h-[500px] mb-6">
          <div className="col-span-6 min-h-0">
            <RealtimeSpeech
              isActive={isActive}
              transcriptions={transcriptions}
              speakerColors={speakerColors}
            />
          </div>

          <div className="col-span-6 min-h-0">
            <Statistics
              speakers={speakers}
              avgTopic={avgTopic}
              avgNovelty={avgNovelty}
              speakerColors={speakerColors}
            />
          </div>
        </div>


        <div className="col-span-12 mb-6">
          <TimelineVisualization
            segments={timeline}
            speakerColors={speakerColors}
          />
        </div>

        <div className="col-span-12">
          <SpeakerAnalysisTable
            speakers={speakers}
            speakerColors={speakerColors}
          />
        </div>
      </main>
    </div>
  );
}
