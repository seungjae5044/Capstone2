import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { StatusIndicator } from './StatusIndicator';

interface HeaderProps {
  isActive: boolean;
  onStart: (topic: string, participants: number) => void;
  onStop: () => void;
  onReport: () => void;
}

export function Header({ isActive, onStart, onStop, onReport }: HeaderProps) {
  const [topic, setTopic] = useState('');
  const [participants, setParticipants] = useState(4);

  return (
    <header className="h-16 bg-white border-b border-[#e5e7eb] px-6 flex items-center justify-between">
      <div className="flex items-center gap-3">
        <div className="w-8 h-8 bg-[#2563eb] rounded-lg flex items-center justify-center">
          <div className="w-4 h-4 bg-white rounded-sm" />
        </div>
        <div>
          <h1 className="text-lg font-semibold text-[#1a1a1a]">MeetingProgram</h1>
          <StatusIndicator isActive={isActive} />
        </div>
      </div>

      <div className="flex items-center gap-3">
        <Input
          placeholder="회의 주제"
          value={topic}
          onChange={(e) => setTopic(e.target.value)}
          className="w-72"
        />
        <Input
          type="number"
          min={2}
          max={12}
          value={participants}
          onChange={(e) => setParticipants(Number(e.target.value))}
          className="w-20"
        />
        <Button
          onClick={() => onStart(topic, participants)}
          disabled={isActive || !topic.trim()}
          className="bg-[#0066cc] hover:bg-[#0052a3]"
        >
          시작
        </Button>
        <Button
          onClick={onStop}
          disabled={!isActive}
          variant="destructive"
        >
          중지
        </Button>
        <Button
          onClick={onReport}
          variant="outline"
        >
          보고서
        </Button>
      </div>
    </header>
  );
}
