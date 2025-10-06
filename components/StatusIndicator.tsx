interface StatusIndicatorProps {
  isActive: boolean;
  currentSpeaker?: string;
}

export function StatusIndicator({ isActive, currentSpeaker }: StatusIndicatorProps) {
  return (
    <div className="flex items-center gap-2">
      <div 
        className={`w-1.5 h-1.5 rounded-full ${
          isActive ? 'bg-[#10B981]' : 'bg-[#9CA3AF]'
        }`}
      />
      <span className="text-sm text-[#6b7280]">
        {isActive ? 'AI 분석 활성' : '대기 중'}
      </span>
      {currentSpeaker && isActive && (
        <span className="text-sm text-[#374151]">- {currentSpeaker}</span>
      )}
    </div>
  );
}