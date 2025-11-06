export interface SpeakerStats {
  speaker_id: string;
  name: string;
  count: number;
  duration: number;
  topic_avg: number;
  novelty_avg: number;
  participation: number;
}

export interface StatsMessage {
  type: 'stats';
  speakers: SpeakerStats[];
  avg_topic: number;
  avg_novelty: number;
}

export interface StatusMessage {
  type: 'status';
  is_active: boolean;
  duration: number;
}

export interface TranscriptionMessage {
  type: 'transcription';
  speaker_id: string;
  speaker_name?: string;
  text: string;
  timestamp: string;
  duration?: number;
  topic_score?: number;
  novelty_score?: number;
  similarity?: number;
}

export interface EvaluationMessage {
  type: 'evaluation';
  speaker_id: string;
  speaker_name?: string;
  text?: string;
  topic_relevance: number;
  novelty: number;
  timestamp: string;
}

export interface StatsUpdateMessage {
  type: 'stats_update';
  overall_stats: {
    total_statements: number;
    avg_topic_relevance: number;
    avg_novelty: number;
  };
  speaker_stats: Record<
    string,
    {
      total_statements: number;
      avg_topic_relevance: number;
      avg_novelty: number;
      [key: string]: number;
    }
  >;
}

export interface SessionStatusMessage {
  type: 'session_status';
  is_active: boolean;
  session_id: string | null;
  topic: string;
}

export interface ReportReadyMessage {
  type: 'report_ready';
  available: boolean;
  path?: string;
  detail?: string;
}

export interface TimelineSegment {
  speaker_id: string;
  speaker_name: string;
  start_time: string;
  end_time: string;
  duration: number;
}

export interface TimelineSegmentMessage {
  type: 'timeline_segment';
  segment: TimelineSegment;
}

export type WebSocketMessage =
  | TranscriptionMessage
  | EvaluationMessage
  | StatsMessage
  | StatsUpdateMessage
  | SessionStatusMessage
  | ReportReadyMessage
  | TimelineSegmentMessage;

export interface TranscriptionEntry {
  id: string;
  speakerId: string;
  speakerName: string;
  text: string;
  timestamp: string;
  duration?: number;
  topicScore?: number;
  noveltyScore?: number;
  similarity?: number;
}

export interface TimelineEntry {
  speakerId: string;
  speakerName: string;
  startTime: string;
  endTime: string;
  duration: number;
}

// Setup and audio device types
export type AudioSource = 'mic' | 'system';

export interface StartRequest {
  topic: string;
  participants: number;
  source: AudioSource;
  device_index?: number;
}

export interface AudioDeviceInfo {
  index: number;
  name: string;
  max_input_channels: number;
  hostapi: string;
}
