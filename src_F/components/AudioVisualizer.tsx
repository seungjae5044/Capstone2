import { useEffect, useState } from 'react';

interface AudioVisualizerProps {
  isActive: boolean;
}

export function AudioVisualizer({ isActive }: AudioVisualizerProps) {
  const [bars, setBars] = useState<number[]>(Array(20).fill(0));

  useEffect(() => {
    if (!isActive) {
      setBars(Array(20).fill(0));
      return;
    }

    const interval = setInterval(() => {
      setBars(Array.from({ length: 20 }, () => Math.random() * 100));
    }, 150);

    return () => clearInterval(interval);
  }, [isActive]);

  return (
    <div className="h-15 bg-[#f1f9ff] rounded-xl p-3 flex items-end justify-center gap-1">
      {bars.map((height, index) => (
        <div
          key={index}
          className="w-1 bg-[#3b82f6] rounded-full transition-all duration-150"
          style={{ height: `${Math.max(4, height * 0.4)}px` }}
        />
      ))}
    </div>
  );
}