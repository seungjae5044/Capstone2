import { useEffect, useMemo, useState } from 'react';
import { Button } from './ui/button';
import { Progress } from './ui/progress';
import { ChevronLeft, ChevronRight } from 'lucide-react';
import type { SpeakerStats } from '../types';

const DEFAULT_SPEAKER_COLORS = ['#2563eb', '#16a34a', '#db2777', '#f59e0b', '#8b5cf6', '#06b6d4', '#f97316', '#84cc16', '#ec4899', '#14b8a6'];

interface StatisticsProps {
  speakers: SpeakerStats[];
  avgTopic: number;
  avgNovelty: number;
  speakerColors: Record<string, string>;
}

export function Statistics({ speakers, avgTopic, avgNovelty, speakerColors }: StatisticsProps) {
  const [currentSpeaker, setCurrentSpeaker] = useState(0);
  const hasSpeakers = speakers.length > 0;

  useEffect(() => {
    if (!speakers.length) {
      setCurrentSpeaker(0);
      return;
    }
    if (currentSpeaker >= speakers.length) {
      setCurrentSpeaker(speakers.length - 1);
    }
  }, [speakers, currentSpeaker]);

  const colorMap = useMemo(() => speakerColors, [speakerColors]);

  const nextSpeaker = () => {
    setCurrentSpeaker((prev) => (prev + 1) % Math.max(1, speakers.length));
  };

  const prevSpeaker = () => {
    setCurrentSpeaker((prev) => (prev - 1 + speakers.length) % Math.max(1, speakers.length));
  };

  const currentData = speakers[currentSpeaker];

  return (
    <div className="bg-white border border-[#e5e7eb] rounded-xl p-6 shadow-sm hover:shadow-md transition-shadow h-full min-h-0 flex flex-col">
      <h2 className="mb-6">통계량</h2>

      <div className="grid grid-cols-10 gap-6 h-full">
        {/* Left side - Charts */}
        <div className="col-span-7">
          <div className="bg-[#f9fafb] rounded-lg h-140 flex flex-col p-[16px]">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-sm text-[#6b7280]">
                {currentData ? `[${currentData.name}] 화자별 주제연관성 & 신규성 막대 그래프` : '화자를 기다리는 중...'}
              </h3>
            </div>

            {currentData ? (
              <div className="flex-1 flex flex-col justify-center space-y-6">
                {/* Topic Relevance Bar */}
                <div>
                  <div className="flex justify-between items-center mb-3">
                    <span className="text-sm text-[#374151]">주제 연관성</span>
                    <span className="text-sm text-[#6b7280]">{currentData.topic_avg.toFixed(1)}</span>
                  </div>
                  <div className="w-full bg-[#e5e7eb] rounded-full h-4">
                    <div
                      className="h-4 rounded-full transition-all duration-500"
                      style={{
                        width: `${(currentData.topic_avg / 10) * 100}%`,
                        backgroundColor: colorMap[currentData.speaker_id] ?? DEFAULT_SPEAKER_COLORS[0],
                      }}
                    />
                  </div>
                </div>

                {/* Novelty Bar */}
                <div>
                  <div className="flex justify-between items-center mb-3">
                    <span className="text-sm text-[#374151]">아이디어 신규성</span>
                    <span className="text-sm text-[#6b7280]">{currentData.novelty_avg.toFixed(1)}</span>
                  </div>
                  <div className="w-full bg-[#e5e7eb] rounded-full h-4">
                    <div
                      className="h-4 rounded-full transition-all duration-500"
                      style={{
                        width: `${(currentData.novelty_avg / 10) * 100}%`,
                        backgroundColor: colorMap[currentData.speaker_id] ?? DEFAULT_SPEAKER_COLORS[1],
                      }}
                    />
                  </div>
                </div>
              </div>
            ) : (
              <div className="flex-1 flex items-center justify-center text-[#6b7280]">
                회의를 시작하면 통계가 표시됩니다
              </div>
            )}

            {/* Pagination */}
            {hasSpeakers && (
              <div className="flex items-center justify-center gap-2 mt-4">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={prevSpeaker}
                  className="w-8 h-8 p-0 border-[#ccc]"
                >
                  <ChevronLeft className="w-4 h-4" />
                </Button>
                <span className="text-sm text-[#6b7280] mx-2">
                  {`${currentSpeaker + 1} / ${speakers.length}`}
                </span>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={nextSpeaker}
                  className="w-8 h-8 p-0 border-[#ccc]"
                >
                  <ChevronRight className="w-4 h-4" />
                </Button>
              </div>
            )}
          </div>
        </div>

        {/* Right side - Metrics and Pie Chart */}
        <div className="col-span-3 space-y-6">
          {/* Average Metrics */}
          <div className="bg-[#f9fafb] rounded-lg p-4">
            <div className="space-y-4">
              <div>
                <div className="flex justify-between items-center mb-2">
                  <span className="text-sm text-[#374151] whitespace-nowrap">주제 연관성</span>
                  <span className="text-sm text-[#6b7280]">{avgTopic.toFixed(1)}</span>
                </div>
                <Progress value={(avgTopic / 10) * 100} className="h-2" />
              </div>

              <div>
                <div className="flex justify-between items-center mb-2">
                  <span className="text-sm text-[#374151] whitespace-nowrap">신규성</span>
                  <span className="text-sm text-[#6b7280]">{avgNovelty.toFixed(1)}</span>
                </div>
                <Progress value={(avgNovelty / 10) * 100} className="h-2" />
              </div>
            </div>
          </div>

          {/* Legend */}
          <div className="bg-[#f9fafb] rounded-lg p-4">
            <h4 className="text-sm text-[#374151] mb-4">화자별 발언 점유율</h4>
            <div className="space-y-2">
              {speakers.map((item) => (
                <div key={item.speaker_id} className="flex items-center gap-2">
                  <div
                    className="w-3 h-3 rounded-full"
                    style={{ backgroundColor: colorMap[item.speaker_id] ?? DEFAULT_SPEAKER_COLORS[0] }}
                  />
                  <span className="text-xs text-[#6b7280]">{item.name}</span>
                  <span className="text-xs text-[#6b7280] ml-auto">{item.participation.toFixed(1)}%</span>
                </div>
              ))}
              {speakers.length === 0 && (
                <div className="text-xs text-[#6b7280] text-center py-4">데이터 없음</div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
