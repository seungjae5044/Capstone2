import React, { useMemo } from 'react';
import type { TimelineEntry } from '../types';

interface TimelineVisualizationProps {
  segments: TimelineEntry[];
  speakerColors: Record<string, string>;
}

function formatClockLabel(date: Date) {
  return `${date.getHours().toString().padStart(2, '0')}:${date.getMinutes().toString().padStart(2, '0')}:${date
    .getSeconds()
    .toString()
    .padStart(2, '0')}`;
}

export function TimelineVisualization({ segments, speakerColors }: TimelineVisualizationProps) {
  // 전체 타임라인 정렬 및 그룹화
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

  const timelineWidthPx = 2000;
  const rowHeight = 36;
  const labelWidth = 128; // 레이블 영역 넓이(px)

  const getPxPosition = (date: Date): number => {
    if (!timelineStart || !timelineDurationMs) return labelWidth;
    const offsetMs = date.getTime() - timelineStart.getTime();
    return labelWidth + (offsetMs / timelineDurationMs) * (timelineWidthPx - labelWidth);
  };

  const ticks = useMemo(() => {
    if (!timelineStart || !timelineEnd) return [];
    const tickCount = 6;
    return Array.from({ length: tickCount }).map((_, i) => {
      const tickTime = new Date(timelineStart.getTime() + (timelineDurationMs * i) / (tickCount - 1));
      return {
        label: formatClockLabel(tickTime),
        leftPx: labelWidth + ((timelineWidthPx - labelWidth) * i) / (tickCount - 1),
      };
    });
  }, [timelineStart, timelineEnd, timelineDurationMs]);

  if (!timelineStart || !timelineEnd) {
    return <div>타임라인 데이터 없음</div>;
  }

  return (
    <div className="bg-white border border-[#e5e7eb] rounded-xl p-6 shadow-sm transition-shadow overflow-hidden">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-semibold">발언 타임라인</h2>
        <span className="text-xs text-[#6b7280]">
          {formatClockLabel(timelineStart)} ~ {formatClockLabel(timelineEnd)}
        </span>
      </div>
      <div className="overflow-x-auto pb-2">
        <div style={{ position: 'relative', minWidth: timelineWidthPx, height: rowHeight * speakers.length + 40 }}>
          {/* 화자 레이블 영역 */}
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
            {ticks.map(tick => (
              <span
                key={tick.label}
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
          {/* 발화 막대 영역 */}
          {speakers.map((speaker, idx) =>
            speaker.segments.map(segment => {
              const leftPx = getPxPosition(segment.start);
              const rightPx = getPxPosition(segment.end);
              const widthPx = Math.max(rightPx - leftPx, 14);
              return (
                <div
                  key={`${segment.speakerId}_${segment.start.toISOString()}_${segment.end.toISOString()}`}
                  title={`${segment.speakerName} (${formatClockLabel(segment.start)} ~ ${formatClockLabel(segment.end)})`}
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
