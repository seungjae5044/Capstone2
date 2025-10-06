import { useState } from 'react';
import { Header } from './components/Header';
import { RealtimeSpeech } from './components/RealtimeSpeech';
import { Statistics } from './components/Statistics';
import { SpeakerAnalysisTable } from './components/SpeakerAnalysisTable';

export default function App() {
  const [isActive, setIsActive] = useState(false);

  const handleStart = () => {
    setIsActive(true);
  };

  const handleStop = () => {
    setIsActive(false);
  };

  return (
    <div className="min-h-screen bg-[#fafbfc]">
      <Header isActive={isActive} onStart={handleStart} onStop={handleStop} />
      
      <main className="max-w-7xl mx-auto px-6 py-6">
        {/* Main Grid Layout */}
        <div className="grid grid-cols-12 gap-6 h-[600px] mb-6">
          {/* Realtime Speech - Left */}
          <div className="col-span-6 row-span-2">
            <RealtimeSpeech isActive={isActive} />
          </div>
          
          {/* Statistics - Right */}
          <div className="col-span-6 row-span-2">
            <Statistics />
          </div>
        </div>
        
        {/* Speaker Analysis Table - Full Width */}
        <div className="col-span-12">
          <SpeakerAnalysisTable />
        </div>
      </main>
    </div>
  );
}