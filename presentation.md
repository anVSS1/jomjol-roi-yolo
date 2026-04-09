Create a professional university project presentation with 9 slides total (8 content slides + 1 final Q&A slide) for a Master's program in Business Intelligence and Big Data Analytics.

Presentation title:
Automated Meter Reading System with Automatic ROI Detection and Socket-Based Distributed Inference

Audience:
Professor, academic evaluators, and jury members.

Duration:
10 to 12 minutes.

Tone:
Technical but clear, confident, and honest about limitations.

Design style:
Modern, clean, data-driven visuals, clear diagrams, and readable charts.

Visual direction (important):

- Use a strict Anthropic website aesthetic (Claude.ai / Anthropic Research style):
  - Highly editorial, academic but modern, brutalist yet warm.
  - Heavy use of negative space, precise grid alignment, and a calm, serious tone.
  - Avoid flashy startup style, neon gradients, and playful icon packs.
- Theme:
  - Light theme only.
  - Background: Warm cream/paper color (#FDFBF7 or #F9F8F6).
  - Primary text: Deep espresso/off-black (#1C1917).
  - Secondary text: Muted stone/taupe (#57534E).
- Accent palette (use consistently across all slides):
  - Primary accent (for key metrics/lines): Muted terracotta / brick red (#D95D39).
  - Secondary accent: Pale sage green or soft ochre.
  - Divider/card stroke: Very thin, faint beige/gray hairline (#E7E5E4).
- Typography:
  - Titles & Large numbers: An elegant, warm serif font (e.g., Tiempos, Recoleta, or Georgia).
  - Body text: Clean, highly legible sans-serif (e.g., Inter, Helvetica, or Roobert).
  - Strong hierarchy: Large elegant serif titles, tight line heights, very readable body copy.
- Slide composition:
  - 16:9 layout, clean grid alignment, generous margins.
  - One key message per slide.
  - Prefer 2-column layouts for "method vs baseline" and "result vs limitation" slides.
  - Use cards with subtle borders instead of heavy shadows.
- Diagram style:
  - Architecture diagram with rounded rectangles, thin connectors, and consistent icon style.
  - Label each pipeline stage clearly (Machine A and Machine B).
- Data visualization style:
  - Use simple bar/line charts with limited colors from the palette above.
  - Highlight only the key metric in accent color; keep others neutral.
  - Avoid 3D charts and unnecessary chart decorations.
- Motion/transitions:
  - Use subtle fades only.
  - No aggressive animations.
- Final quality rule:
  - The deck must look like a polished technical research presentation, not a generic template.

Output format requirements:

- For each slide, provide:
  - Slide title
  - 3 to 5 concise bullet points
  - Speaker notes script (70 to 110 words)
  - Suggested visual (diagram, screenshot, table, or chart)
- Add a final Q&A slide.
- DO NOT invent or confidently state any fake numerical data that is not explicitly stated in the prompt.
- Use plain English and avoid hype.

Story arc to follow:

1. Introduction & The LBC Matrix Overview
2. L-Axis (Literature): The Manual ROI Bottleneck in Edge Systems
3. B-Axis (Business Need): The Scalability Problem
4. C-Axis (Contribution): Distributed YOLO+CNN Architecture
5. C-Axis (Component Results): YOLO Detection & CNN Classification
6. State of the Art Benchmark (Positioning)
7. Experimental Results & Error Analysis
8. Conclusion
9. Q&A

Mandatory slide-by-slide technical content guidelines:

Slide 1: Title Slide

- Title: "Automated Meter Reading System with Automatic ROI Detection and Socket-Based Distributed Inference"
- Subtitle: "Following the LBC Matrix"

Slide 2: L-Axis (Literature) - The Baseline Problem

- Original baseline (jomjol AI-on-the-Edge-Device) required manual ROI selection (digit strip boundaries) via a web UI.
- If the camera shifts, the ROI must be recalibrated manually.
- This creates a deployment bottleneck for real-world edge installations.
- Visual requirement: Use the screenshot showing the manual Jomjol Web UI configuration (`jomjol roi .png`).

Slide 3: B-Axis (Business Need) - The Scalability Bottleneck

- Utility companies cannot afford human intervention for hundreds of thousands of installed meters.
- Large deployments require autonomy, low maintenance, and robustness to small camera shifts.
- The core need is to eliminate manual UI configuration and create a fully autonomous reading pipeline.

Slide 4: C-Axis (Contribution) - Distributed Architecture

- The system uses a two-machine setup connected via TCP sockets.
- Machine A (Client): captures the image, runs YOLOv8n ROI extraction, and sends cropped image bytes.
- Machine B (Server): receives the crop, applies OpenCV segmentation, runs CNN classification, and reconstructs the final reading.
- The architecture separates lightweight acquisition from heavier inference logic.
- Visual requirement: Include a clean architecture flowchart diagram.

Slide 5: C-Axis - Component Validation (YOLO & CNN)

- YOLO achieved 100% ROI detection success on 50 test images.
- YOLO validation performance reached 99.34% mAP@50 and 75.62% mAP@50-95.
- The custom CNN was trained on 9,960 digit crops.
- The CNN used 7,968 training samples and 1,992 test samples.
- Final CNN hold-out accuracy reached 99.00%.
- Visual requirement: Show the YOLO bounding box crop (`yolo crop.png`) and the CNN debug panel (`cnn detection.png`) showing the reading "05866".

Slide 6: State of the Art Benchmark

- Compare the proposed system against:
  - Traditional CV pipelines
  - Legacy Edge (Jomjol)
  - Laroca et al. (server-oriented deep learning AMR)
  - Salomon et al. (dial meter deep learning approach)
- Compare along these dimensions:
  - System
  - ROI Localization
  - Autonomy
  - Scalability
- The key positioning message: the proposed solution bridges the gap by combining high autonomy with high scalability.
- Visual requirement: Create a clean 4-row comparison table.

Slide 7: End-to-End Results & Error Analysis

- The complete pipeline achieved 72.00% exact full-reading match on 50 local test images.
- Correct text-length prediction reached 76.00%.
- The main limitation is deterministic OpenCV under-segmentation.
- Glare and reflections can erase visible gaps between digits.
- Example failure: the real reading "21931" was incorrectly reconstructed as "43".
- Visual requirement: Show the specific error case image (`cnn error.jpg`).

Slide 8: Conclusion

- The proposed system fully removes the manual ROI bottleneck found in legacy edge workflows.
- It achieves strong component-level performance: 100% ROI detection success and 99.00% digit classification accuracy.
- The distributed socket-based design makes the pipeline more practical for scalable and edge-compatible deployment.
- The main limitation remains segmentation robustness under difficult lighting conditions.
- Future work should move from deterministic segmentation toward sequence-based OCR or end-to-end recognition.

Slide 9: Q&A

- Title: "Q&A"
- Keep this slide visually minimal and consistent with the rest of the deck.

CNN notebook evidence (must be respected in generated content):

- Use only these CNN numbers:
  - 9,960 total digit crops
  - 7,968 training samples
  - 1,992 test samples
  - 99.00% final test accuracy
  - macro avg = 0.99
  - weighted avg = 0.99
- Training run details:
  - early stopping at epoch 124
  - best restored checkpoint at epoch 104
- Do not mention old CNN metrics such as 97.70% or any MNIST comparison because they are outdated for this run.

Important factual constraints:

- Do not claim that the full pipeline achieves 99% end-to-end reading accuracy.
- 99.00% refers only to CNN digit classification accuracy on the hold-out test set.
- 72.00% is the exact full-reading match rate for the complete pipeline.
- Keep the conclusion honest about segmentation limitations.