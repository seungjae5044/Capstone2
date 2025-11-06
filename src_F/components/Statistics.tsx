import { useState } from 'react';
import { Button } from './ui/button';
import { Progress } from './ui/progress';
import { ChevronLeft, ChevronRight } from 'lucide-react';

const speakerData = [
  { name: '김팀장', topic: 85, novelty: 75 },
  { name: '이대리', topic: 60, novelty: 80 },
  { name: '박과장', topic: 70, novelty: 65 },
  { name: '최사원', topic: 55, novelty: 90 }
];

const pieChartData = [
  { name: '김팀장', value: 30, color: '#2563eb' },
  { name: '이대리', value: 25, color: '#16a34a' },
  { name: '박과장', value: 20, color: '#db2777' },
  { name: '최사원', value: 25, color: '#f59e0b' }
];

export function Statistics() {
  const [currentSpeaker, setCurrentSpeaker] = useState(0);

  const nextSpeaker = () => {
    setCurrentSpeaker((prev) => (prev + 1) % speakerData.length);
  };

  const prevSpeaker = () => {
    setCurrentSpeaker((prev) => (prev - 1 + speakerData.length) % speakerData.length);
  };

  const currentData = speakerData[currentSpeaker];

  return (
    <div className="bg-white border border-[#e5e7eb] rounded-xl p-6 shadow-sm hover:shadow-md transition-shadow h-full">
      <h2 className="mb-6">통계량</h2>
      
      <div className="grid grid-cols-10 gap-6 h-full">
        {/* Left side - Charts */}
        <div className="col-span-7">
          {/* Bar Chart Section - Match right side height */}
          <div className="bg-[#f9fafb] rounded-lg h-140 flex flex-col p-[16px]">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-sm text-[#6b7280]">
                [{currentData.name}] 화자별 주제연관성 & 신규성 막대 그래프
              </h3>
            </div>
            
            <div className="flex-1 flex flex-col justify-center space-y-6">
              {/* Topic Relevance Bar */}
              <div>
                <div className="flex justify-between items-center mb-3">
                  <span className="text-sm text-[#374151]">주제 연관성</span>
                  <span className="text-sm text-[#6b7280]">{currentData.topic}%</span>
                </div>
                <div className="w-full bg-[#e5e7eb] rounded-full h-4">
                  <div 
                    className="bg-[#2563eb] h-4 rounded-full transition-all duration-500"
                    style={{ width: `${currentData.topic}%` }}
                  />
                </div>
              </div>
              
              {/* Novelty Bar */}
              <div>
                <div className="flex justify-between items-center mb-3">
                  <span className="text-sm text-[#374151]">아이디어 신규성</span>
                  <span className="text-sm text-[#6b7280]">{currentData.novelty}%</span>
                </div>
                <div className="w-full bg-[#e5e7eb] rounded-full h-4">
                  <div 
                    className="bg-[#16a34a] h-4 rounded-full transition-all duration-500"
                    style={{ width: `${currentData.novelty}%` }}
                  />
                </div>
              </div>
            </div>
            
            {/* Pagination */}
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
                {currentSpeaker + 1} / {speakerData.length}
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
          </div>
        </div>
        
        {/* Right side - Metrics and Pie Chart */}
        <div className="col-span-3 space-y-6">
          {/* Average Metrics */}
          <div className="bg-[#f9fafb] rounded-lg p-4">
            <div className="space-y-4">
              <div>
                <div className="flex justify-between items-center mb-2">
                  <span className="text-sm text-[#374151]">주제 연관성 (평균)</span>
                  <span className="text-sm text-[#6b7280]">67.5%</span>
                </div>
                <Progress value={67.5} className="h-2" />
              </div>
              
              <div>
                <div className="flex justify-between items-center mb-2">
                  <span className="text-sm text-[#374151]">신규성 (평균)</span>
                  <span className="text-sm text-[#6b7280]">77.5%</span>
                </div>
                <Progress value={77.5} className="h-2" />
              </div>
            </div>
          </div>
          
          {/* Pie Chart */}
          <div className="bg-[#f9fafb] rounded-lg p-4">
            <h4 className="text-sm text-[#374151] mb-4">화자별 발언 점유율</h4>
            
            {/* Simple pie chart representation */}
            <div className="relative w-24 h-24 mx-auto mb-4">
              <svg className="w-full h-full transform -rotate-90" viewBox="0 0 100 100">
                <circle
                  cx="50"
                  cy="50"
                  r="40"
                  fill="none"
                  stroke="#2563eb"
                  strokeWidth="20"
                  strokeDasharray="75.4 251.3"
                  strokeDashoffset="0"
                />
                <circle
                  cx="50"
                  cy="50"
                  r="40"
                  fill="none"
                  stroke="#16a34a"
                  strokeWidth="20"
                  strokeDasharray="62.8 251.3"
                  strokeDashoffset="-75.4"
                />
                <circle
                  cx="50"
                  cy="50"
                  r="40"
                  fill="none"
                  stroke="#db2777"
                  strokeWidth="20"
                  strokeDasharray="50.3 251.3"
                  strokeDashoffset="-138.2"
                />
                <circle
                  cx="50"
                  cy="50"
                  r="40"
                  fill="none"
                  stroke="#f59e0b"
                  strokeWidth="20"
                  strokeDasharray="62.8 251.3"
                  strokeDashoffset="-188.5"
                />
              </svg>
            </div>
            
            {/* Legend */}
            <div className="space-y-2">
              {pieChartData.map((item) => (
                <div key={item.name} className="flex items-center gap-2">
                  <div 
                    className="w-3 h-3 rounded-full"
                    style={{ backgroundColor: item.color }}
                  />
                  <span className="text-xs text-[#6b7280]">{item.name}</span>
                  <span className="text-xs text-[#6b7280] ml-auto">{item.value}%</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}