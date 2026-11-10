# Project-Bouldering
Video-to-analytics pipeline for bouldering: pose keypoints, climbing metrics, and skeleton-overlay feedback.
Project Boulder turns raw climbing footage into reliable, queryable data about how a climber moves. Videos of bouldering attempts are ingested with metadata (grade, wall angle, sent or fell), run through pose estimation, and stored as keypoint time series. Transformations compute movement metrics such as arm bend, foot readjustments, hesitation time and hip-path smoothness, and data-quality checks flag low-confidence frames. The output is a feedback video with a skeleton overlay that highlights inefficient moves.

Roadmap: v1 single camera + rule-based metrics → v2 multi-angle 3D reconstruction → v3 learned models and physics simulation.
