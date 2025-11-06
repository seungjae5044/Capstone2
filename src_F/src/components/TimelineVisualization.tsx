import React, { useMemo } from 'react';
import type { TimelineEntry } from '../types';

interface TimelineVisualizationProps {
  segments: TimelineEntry[];
  speakerColors: Record<string, string>;
}

function formatShortClockLabel(date: Date) {
  // '15:13' 처럼 간단하게 반환
  return `${date.getHours().toString().padStart(2, '0')}:${date.getMinutes().toString().padStart(2, '0')}`;
}

export function TimelineVisualization({ segments, speakerColors }: TimelineVisualizationProps) {
  // ... 타임라인 정렬 및 그룹화는 기존 그대로

  const { timelineStart, timelineEnd, speakers, timelineDurationMs } = useMemo(() => {
    if (!segments || segments.length === 0) {
      return { timelineStart: null, timelineEnd: null, speakers: [], timelineDurationMs: 0 };
    }

    let minStart = new Date(segments[0].startTime);
    let maxEnd = new Date(segments[0].endTime);

    const map = new Map<string, { name: string; segments: any[] }>();

    for (const seg of segments) {
      const start = new Date(seg.startTime);
      const end = new Date(seg.endTime);
      if (start < minStart) minStart = start;
      if (end > maxEnd) maxEnd = end;

      const speakerId = seg.speakerId;
      if (!map.has(speakerId))
        map.set(speakerId, { name: seg.speakerName, segments: [] });
      map.get(speakerId)!.segments.push({ ...seg, start, end });
    }

    return {
      timelineStart: minStart,
      timelineEnd: maxEnd,
      speakers: Array.from(map.entries()).map(([id, { name, segments }]) => ({
        speakerId: id,
        speakerName: name,
        segments,
      })),
      timelineDurationMs: maxEnd.getTime() - minStart.getTime(),
    };
  }, [segments]);

  const rowHeight = 36;
  const labelWidth = 128;
  const pxPerSecond = 3;

  // 타임라인 전체 픽셀 크기 계산
  const timelineWidthPx = useMemo(() => {
    if (!timelineDurationMs) return 800;
    const durationSeconds = timelineDurationMs / 1000;
    return labelWidth + (durationSeconds * pxPerSecond);
  }, [timelineDurationMs]);

  const getPxPosition = (date: Date): number => {
    if (!timelineStart || !timelineDurationMs) return labelWidth;
    const offsetMs = date.getTime() - timelineStart.getTime();
    return labelWidth + (offsetMs / timelineDurationMs) * (timelineWidthPx - labelWidth);
  };

  // '틱(눈금선)' 기본 생성
  const ticks = useMemo(() => {
    if (!timelineStart || !timelineEnd) return [];
    const durationSeconds = timelineDurationMs / 1000;
    // 기본 tickInterval: 10초
    let tickIntervalSeconds = 10;
    if (durationSeconds < 30) tickIntervalSeconds = 5;
    if (durationSeconds > 120) tickIntervalSeconds = 30;

    const tickCount = Math.ceil(durationSeconds / tickIntervalSeconds) + 1;

    return Array.from({ length: tickCount }).map((_, i) => {
      const tickTime = new Date(timelineStart.getTime() + (i * tickIntervalSeconds * 1000));
      const offsetSeconds = (tickTime.getTime() - timelineStart.getTime()) / 1000;
      return {
        label: formatShortClockLabel(tickTime), // 라벨을 간편하게 표현
        leftPx: labelWidth + (offsetSeconds * pxPerSecond),
      };
    });
  }, [timelineStart, timelineEnd, timelineDurationMs]);

  // 눈금 라벨 겹침 방지 (최소 간격 설정) 예시
  const MIN_LABEL_GAP_PX = 45;
  const filteredTicks = ticks.filter((tick, i, arr) => {
    if (i === 0) return true;
    return (tick.leftPx - arr[i - 1].leftPx) >= MIN_LABEL_GAP_PX;
  });

  if (!timelineStart || !timelineEnd) {
    return (
      <div className="bg-white border border-[#e5e7eb] rounded-xl p-6 shadow-sm transition-shadow">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold">발언 타임라인</h2>
        </div>
        <div className="text-center text-[#6b7280] py-12">
          회의를 시작하면 발언 타임라인이 표시됩니다.
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white border border-[#e5e7eb] rounded-xl p-6 shadow-sm transition-shadow overflow-hidden">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-semibold">발언 타임라인</h2>
        <span className="text-xs text-[#6b7280]">
          {formatShortClockLabel(timelineStart)} ~ {formatShortClockLabel(timelineEnd)}
        </span>
      </div>
      <div className="overflow-x-auto pb-2 scrollbar-thin">
        <div style={{ position: 'relative', minWidth: timelineWidthPx, height: rowHeight * speakers.length + 40 }}>
          {speakers.map((speaker, idx) => (
            <div
              key={`label:${speaker.speakerId}`}
              style={{
                position: 'absolute',
                left: 0,
                top: 28 + idx * rowHeight,
                height: rowHeight,
                width: labelWidth,
                display: 'flex',
                alignItems: 'center',
                gap: 8,
                fontSize: 14,
                zIndex: 10,
                background: 'white',
              }}
            >
              <span
                style={{
                  width: 14,
                  height: 14,
                  borderRadius: '50%',
                  backgroundColor: speakerColors[speaker.speakerId] ?? '#2563eb',
                }}
              />
              {speaker.speakerName}
            </div>
          ))}
          {/* 눈금선 */}
          <div style={{ position: 'absolute', left: labelWidth, top: 0, width: timelineWidthPx - labelWidth, height: 20 }}>
            {filteredTicks.map(tick => (
              <span
                key={tick.label + tick.leftPx}
                style={{
                  position: 'absolute',
                  left: tick.leftPx,
                  transform: 'translateX(-50%)',
                  fontSize: 12,
                  color: '#475569',
                  userSelect: 'none',
                }}
              >
                {tick.label}
              </span>
            ))}
          </div>
          {/* 발화 막대 */}
          {speakers.map((speaker, idx) =>
            speaker.segments.map(segment => {
              const leftPx = getPxPosition(segment.start);
              const rightPx = getPxPosition(segment.end);
              const widthPx = Math.max(rightPx - leftPx, 14);
              return (
                <div
                  key={`${segment.speakerId}${segment.start.toISOString()}${segment.end.toISOString()}`}
                  title={`${segment.speakerName} (${formatShortClockLabel(segment.start)} ~ ${formatShortClockLabel(segment.end)})`}
                  style={{
                    position: 'absolute',
                    left: leftPx,
                    top: 28 + idx * rowHeight,
                    height: rowHeight - 8,
                    width: widthPx,
                    backgroundColor: speakerColors[speaker.speakerId] ?? '#2563eb',
                    borderRadius: 5,
                    color: 'white',
                    fontSize: 13,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    padding: '0 8px',
                    zIndex: 2,
                  }}
                >
                  {segment.speakerName}
                </div>
              );
            })
          )}
        </div>
      </div>
    </div>
  );
}
