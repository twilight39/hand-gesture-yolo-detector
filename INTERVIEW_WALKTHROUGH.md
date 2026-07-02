# Interview Walkthrough: Hand Gesture YOLO Detector

A 5-minute summary of the project for technical interviews.

## 1. Problem

Build a real-time, edge-friendly hand-gesture recognizer that demonstrates end-to-end ML skills: data preparation, model fine-tuning, evaluation, export, and deployment.

## 2. Solution

- Fine-tuned YOLOv8n on a 5,000-image subset of HaGRID v2 512px
- 5 classes: open_palm, closed_fist, thumbs_up, peace_sign, no_gesture
- Trained at 320 px for fast CPU/edge inference
- Exported to ONNX and CoreML

## 3. Key Metrics

| Metric | Value |
|--------|-------|
| mAP50 | 0.990 |
| mAP50-95 | 0.806 |
| mAP75 | 0.931 |
| Precision | 0.984 |
| Recall | 0.976 |
| ONNX CPU FPS | 71 |
| PyTorch MPS FPS | 84.5 |

## 4. Key Design Decisions

**Why YOLOv8n?**  
Fastest, smallest variant. Hand gestures are simple shapes; YOLOv8n has enough capacity while keeping latency low.

**Why split by user_id?**  
HaGRID provides a person ID for each image. Splitting by user prevents the same person from appearing in both train and test, giving a more realistic accuracy estimate.

**Why no custom augmentation?**  
YOLOv8's built-in augmentation plus HaGRID's natural diversity (different people, lighting, backgrounds) was sufficient. Custom augmentation risked distorting hand shapes without clear benefit.

**Why 320 px input?**  
Speed. Edge inference matters more than marginal accuracy for this use case. Retraining at 640 px is the obvious next step if accuracy needs dominate.

## 5. What I Learned

- How to prepare a real dataset for YOLO format
- How to avoid data leakage when splitting
- How to export and benchmark models for edge deployment
- How to work around an MPS bug in Ultralytics' loss preprocessing

## 6. Limitations

- `no_gesture` is the weakest class because idle hands look similar to other gestures
- Does not handle severe occlusion or multiple overlapping hands
- Demo environment has good lighting; performance may degrade in poor conditions
- Not yet tested on true embedded hardware (Raspberry Pi / Jetson)

## 7. Next Steps

- Deploy on Raspberry Pi or Jetson with ONNX Runtime
- Add a lightweight tracker to reduce per-frame inference cost
- Expand to more gestures or hand-sign alphabets
