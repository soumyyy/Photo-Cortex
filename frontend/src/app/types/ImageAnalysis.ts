export interface ImageMetadata {
  date_taken: string | null;
  camera_make: string | null;
  camera_model: string | null;
  focal_length: string | null;
  exposure_time: string | null;
  f_number: string | null;
  iso: string | null;
  dimensions: string;
  format: string;
  file_size: string;
  gps: {
    latitude: number;
    longitude: number;
  } | null;
}

export interface ImageAnalysis {
  filename: string;
  metadata: ImageMetadata;
  faces: Face[];
  objects: string[];
  scene_classification: {
    scene_type: string;
    confidence: number;
  } | null;
  text_recognition?: {
    text_detected: boolean;
    text_blocks: Array<{
      text: string;
      confidence: number;
      bbox: {
        x_min: number;
        y_min: number;
        x_max: number;
        y_max: number;
      };
    }>;
    total_confidence: number;
    categories: string[];
    raw_text: string;
    language: string;
  };
}

export interface Face {
  bbox: number[];
  score: number;
  face_id?: string;
  face_image?: string;
  attributes?: {
    age: string;
    gender: string;
    smile_intensity: number;
    eye_status: string;
    eye_metrics: {
      left_ear: number;
      right_ear: number;
    };
    landmarks: number[][];
  };
}