import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from './ui/table';

const tableData = [
  {
    name: '김팀장',
    speechCount: 15,
    duration: '8분 32초',
    topicRelevance: '85%',
    novelty: '75%',
    participation: '30%'
  },
  {
    name: '이대리',
    speechCount: 12,
    duration: '6분 45초',
    topicRelevance: '60%',
    novelty: '80%',
    participation: '25%'
  },
  {
    name: '박과장',
    speechCount: 10,
    duration: '5분 18초',
    topicRelevance: '70%',
    novelty: '65%',
    participation: '20%'
  },
  {
    name: '최사원',
    speechCount: 13,
    duration: '7분 12초',
    topicRelevance: '55%',
    novelty: '90%',
    participation: '25%'
  }
];

export function SpeakerAnalysisTable() {
  return (
    <div className="bg-white border border-[#e5e7eb] rounded-xl p-6 shadow-sm hover:shadow-md transition-shadow">
      <h2 className="mb-6">화자별 분석 (예시)</h2>
      
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
            {tableData.map((row, index) => (
              <TableRow key={row.name} className="hover:bg-[#f9fafb]">
                <TableCell className="text-[#1a1a1a]">{row.name}</TableCell>
                <TableCell className="text-[#6b7280]">{row.speechCount}</TableCell>
                <TableCell className="text-[#6b7280]">{row.duration}</TableCell>
                <TableCell className="text-[#6b7280]">{row.topicRelevance}</TableCell>
                <TableCell className="text-[#6b7280]">{row.novelty}</TableCell>
                <TableCell className="text-[#6b7280]">{row.participation}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </div>
    </div>
  );
}