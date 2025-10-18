import React, { useCallback, useEffect, useMemo, useState } from 'react';
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
  const timelineData = useMemo(() => {
    if (segments.length === 0) {
      return { items: [], totalMs: 0, start: null as Date | null, end: null as Date | null, legends: [] };
    }

    const parsed = segments
      .map(segment => {
        const start = new Date(segment.startTime);
        const end = new Date(segment.endTime);
        if (Number.isNaN(start.getTime()) || Number.isNaN(end.getTime())) {
          return null;
        }
        return {
          ...segment,
          start,
          end,
        };
      })
      .filter(Boolean) as Array<TimelineEntry & { start: Date; end: Date }>;

    if (parsed.length === 0) {
      return { items: [], totalMs: 0, start: null as Date | null, end: null as Date | null, legends: [] };
    }

    const start = parsed.reduce(
      (min, item) => (item.start < min ? item.start : min),
      parsed[0].start,
    );
    const end = parsed.reduce(
      (max, item) => (item.end > max ? item.end : max),
      parsed[0].end,
    );
    const totalMs = Math.max(end.getTime() - start.getTime(), 1);
    const items = parsed.map(item => {
      const left = ((item.start.getTime() - start.getTime()) / totalMs) * 100;
      const width = Math.max(((item.end.getTime() - item.start.getTime()) / totalMs) * 100, 1);
      return {
        speakerId: item.speakerId,
        speakerName: item.speakerName,
        startDate: item.start,
        endDate: item.end,
        left,
        width,
      };
    });

    const legendMap = new Map<
      string,
      { speakerId: string; speakerName: string; firstStart: Date; lastEnd: Date; totalSeconds: number; segments: number }
    >();

    for (const item of items) {
      const entry = legendMap.get(item.speakerId);
      const durationSeconds = (item.endDate.getTime() - item.startDate.getTime()) / 1000;
      if (entry) {
        entry.totalSeconds += durationSeconds;
        entry.segments += 1;
        if (item.startDate < entry.firstStart) {
          entry.firstStart = item.startDate;
        }
        if (item.endDate > entry.lastEnd) {
          entry.lastEnd = item.endDate;
        }
      } else {
        legendMap.set(item.speakerId, {
          speakerId: item.speakerId,
          speakerName: item.speakerName,
          firstStart: item.startDate,
          lastEnd: item.endDate,
          totalSeconds: durationSeconds,
          segments: 1,
        });
      }
    }

    const legends = Array.from(legendMap.values()).map(entry => ({
      ...entry,
      totalSeconds: Math.max(entry.totalSeconds, 0),
    }));

    return { items, totalMs, start, end, legends };
  }, [segments]);

  return (
    <div className="bg-white border border-[#e5e7eb] rounded-xl p-6 shadow-sm hover:shadow-md transition-shadow overflow-hidden">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-semibold">발언 타임라인</h2>
        {timelineData.start && timelineData.end && (
          <span className="text-xs text-[#6b7280]">
            {formatClockLabel(timelineData.start)} ~ {formatClockLabel(timelineData.end)}
          </span>
        )}
      </div>

      {timelineData.items.length === 0 ? (
        <div className="text-center text-[#6b7280] py-10">타임라인 데이터가 없습니다.</div>
      ) : (
        <div className="space-y-4">
          <div className="relative h-24 rounded-lg border border-dashed border-[#d1d5db] bg-[#f9fafb]">
            {timelineData.items.map(item => (
              <div
                key={`${item.speakerId}-${item.startDate.toISOString()}`}
                className="absolute top-3 h-7 rounded-md px-3 flex items-center text-xs font-medium text-white"
                style={{
                  left: `${item.left}%`,
                  width: `${item.width}%`,
                  backgroundColor: speakerColors[item.speakerId] ?? '#2563eb',
                }}
                title={`${item.speakerName} (${formatClockLabel(item.startDate)} ~ ${formatClockLabel(item.endDate)})`}
              >
                <span className="truncate">{item.speakerName}</span>
              </div>
            ))}
          </div>

          <div className="flex flex-wrap gap-3">
            {timelineData.legends?.map(item => (
              <div
                key={`legend-${item.speakerId}`}
                className="flex items-center gap-2 text-xs text-[#4b5563]"
              >
                <span
                  className="inline-block size-2.5 rounded-full"
                  style={{ backgroundColor: speakerColors[item.speakerId] ?? '#2563eb' }}
                />
                <span className="font-medium">{item.speakerName}</span>
                <span className="text-[#9ca3af]">
                  {item.segments}회 / {(item.totalSeconds ?? 0).toFixed(1)}초
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
