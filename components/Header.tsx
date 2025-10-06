import { useState } from 'react';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { StatusIndicator } from './StatusIndicator';

interface HeaderProps {
  isActive: boolean;
  onStart: () => void;
  onStop: () => void;
}

export function Header({ isActive, onStart, onStop }: HeaderProps) {
  const [meetingTopic, setMeetingTopic] = useState('');
  const [participantCount, setParticipantCount] = useState(4);

  const handleStart = () => {
    onStart();
  };

  const handleStop = () => {
    onStop();
  };

  const handleReport = () => {
    console.log('Generate report');
  };

  return (
    <header className="h-16 bg-white border-b border-[#e5e7eb] px-6 flex items-center justify-between">
      <div className="flex items-center gap-3">
        <div className="w-8 h-8 bg-[#2563eb] rounded-lg flex items-center justify-center">
          <div className="w-4 h-4 bg-white rounded-sm"></div>
        </div>
        <div className="flex flex-col">
          <h1 className="text-lg font-semibold text-[#1a1a1a]">MeetingProgram</h1>
          <StatusIndicator isActive={isActive} />
        </div>
      </div>
      
      <div className="flex items-center gap-3">
        <Input
          placeholder="회의 주제"
          value={meetingTopic}
          onChange={(e) => setMeetingTopic(e.target.value)}
          className="w-72 bg-white border border-[#e5e7eb]"
        />
        <Input
          type="number"
          min={2}
          max={12}
          value={participantCount}
          onChange={(e) => setParticipantCount(Number(e.target.value))}
          className="w-20 bg-white border border-[#e5e7eb]"
        />
        <Button
          onClick={handleStart}
          disabled={isActive}
          className="bg-[#0066cc] hover:bg-[#0052a3] text-white"
        >
          시작
        </Button>
        <Button
          onClick={handleStop}
          disabled={!isActive}
          variant="destructive"
          className="bg-[#dc2626] hover:bg-[#b91c1c] text-white disabled:bg-[#e5e7eb] disabled:text-[#9ca3af]"
        >
          중지
        </Button>
        <Button
          onClick={handleReport}
          variant="outline"
          className="bg-white border border-[#d1d1d1] text-[#374151] hover:bg-[#f9fafb]"
        >
          보고서
        </Button>
      </div>
    </header>
  );
}