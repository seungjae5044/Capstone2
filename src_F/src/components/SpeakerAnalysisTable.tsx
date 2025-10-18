import React, { useCallback, useEffect, useMemo, useState } from 'react';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from './ui/table';
import type { SpeakerStats } from '../types';

interface SpeakerAnalysisTableProps {
  speakers: SpeakerStats[];
  speakerColors: Record<string, string>;
}

const DEFAULT_COLORS = ['#2563eb', '#16a34a', '#db2777', '#f59e0b', '#8b5cf6'];

export function SpeakerAnalysisTable({ speakers, speakerColors }: SpeakerAnalysisTableProps) {
  const formatDuration = (seconds: number) => {
    const value = Number.isFinite(seconds) ? seconds : 0;
    const m = Math.floor(value / 60);
    const s = Math.floor(value % 60);
    return `${m}분 ${s}초`;
  };

  return (
    <div className="bg-white border border-[#e5e7eb] rounded-xl p-6 shadow-sm hover:shadow-md transition-shadow">
      <h2 className="mb-6">화자별 분석</h2>

      <div className="rounded-lg border border-[#e5e7eb] overflow-hidden">
        <Table>
          <TableHeader>
            <TableRow className="bg-[#f9fafb]">
              <TableHead className="text-[#374151] border-b border-[#e5e7eb]">이름</TableHead>
              <TableHead className="text-[#374151] border-b border-[#e5e7eb]">발언 수</TableHead>
              <TableHead className="text-[#374151] border-b border-[#e5e7eb]">시간</TableHead>
              <TableHead className="text-[#374151] border-b border-[#e5e7eb]">주제 연관성</TableHead>
              <TableHead className="text-[#374151] border-b border-[#e5e7eb]">아이디어 신규성</TableHead>
              <TableHead className="text-[#374151] border-b border-[#e5e7eb]">참여 비율</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {speakers.map((speaker) => (
              <TableRow key={speaker.speaker_id} className="hover:bg-[#f9fafb]">
                <TableCell className="text-[#1a1a1a]">
                  <span className="inline-flex items-center gap-2">
                    <span
                      className="inline-block size-2.5 rounded-full"
                      style={{ backgroundColor: speakerColors[speaker.speaker_id] ?? DEFAULT_COLORS[0] }}
                    />
                    {speaker.name}
                  </span>
                </TableCell>
                <TableCell className="text-[#6b7280]">{speaker.count}</TableCell>
                <TableCell className="text-[#6b7280]">{formatDuration(speaker.duration)}</TableCell>
                <TableCell className="text-[#6b7280]">{speaker.topic_avg.toFixed(1)}</TableCell>
                <TableCell className="text-[#6b7280]">{speaker.novelty_avg.toFixed(1)}</TableCell>
                <TableCell className="text-[#6b7280]">{speaker.participation.toFixed(1)}%</TableCell>
              </TableRow>
            ))}
            {speakers.length === 0 && (
              <TableRow>
                <TableCell colSpan={6} className="text-center text-[#6b7280] py-8">
                  회의를 시작하면 화자별 분석이 표시됩니다
                </TableCell>
              </TableRow>
            )}
          </TableBody>
        </Table>
      </div>
    </div>
  );
}
