# Interview & Resume Guide

## A. Architecture / data flow
1. **Camera** (`core/camera.py`) – OpenCV `VideoCapture`, explicit open/read/release, errors become `CameraError`.
2. **Preprocessing** – validate frame (8-bit HxWx3), shrink by `frame_scale` (speed), convert **BGR→RGB** (OpenCV is BGR, dlib expects RGB).
3. **Face detection** – HOG + linear SVM returns boxes; boxes are scaled back to full-resolution coordinates.
4. **Face embedding** – dlib aligns the face from landmarks and a pretrained CNN outputs a 128-D vector.
5. **Similarity** – Euclidean distance from the query to every cached stored embedding (one vectorised NumPy call).
6. **Identity matching** – nearest sample; `distance <= threshold` → known, else "Unknown".
7. **Attendance validation** – cooldown check (memory cache, then DB) so one person is marked once per window.
8. **Database** – insert into `attendance`; GUI shows the result. Embeddings live in `face_embeddings`, loaded once into memory at start.

## B. ML in simple words (for the interview)
- A **face embedding** is a list of 128 numbers describing a face so that "same person = nearby points, different people = far apart" (like GPS coordinates for identity).
- **Feature extraction** = the pretrained CNN turning pixels into those numbers.
- **Pretrained model / transfer**: someone else trained the network on millions of faces; we reuse it, so we need only a few photos per person and no GPU or training. We do **not** train a network.
- **Recognition = nearest neighbour + threshold.** Measure distance to stored faces; accept only if it is small enough. The threshold is the security/convenience dial: lower → stricter (more "Unknown" for real users), higher → more false matches.
- **Why not compare images?** Lighting/pose/size change every pixel; embeddings are designed to ignore those.
- **Honest limits:** no liveness check, bias/accuracy differences across groups and conditions, false accepts and false rejects exist.

## C. Interview questions and answers (40)
**Python**
1. *Why a custom exception hierarchy?* A base `AppError` lets the GUI catch every expected failure and show a friendly message, while unexpected exceptions are logged separately.
2. *Why dataclasses (`Settings`, `User`, `MatchResult`)?* Less boilerplate, immutability with `frozen=True`, built-in equality, easy `replace()`.
3. *How did you make the code testable?* Dependency injection: detector/encoder take a `backend` callable, `Camera` takes a capture factory, `AttendanceManager` takes a clock, so tests use fakes with no webcam or real model.
4. *Why a thread for recognition?* Detection/encoding takes tens to hundreds of ms; on the Tk main thread the UI would freeze. A worker thread produces frames; the GUI only polls.

**OpenCV / Computer Vision**
5. *Why BGR→RGB?* OpenCV stores channels as BGR; dlib/PIL expect RGB; skipping the conversion swaps red/blue and hurts accuracy.
6. *Why resize before detection?* Detector cost grows with pixel count; a 0.25× frame is ~16× fewer pixels. Boxes are scaled back by 1/scale. Trade-off: very small/far faces are missed.
7. *How do you measure blur?* Variance of the Laplacian of the face crop: sharp images have strong edges (high variance), blurry ones low. The threshold is configurable and camera-dependent.
8. *What does `INTER_AREA` do?* A resampling mode best suited to shrinking images (less aliasing).
9. *Why does the camera use `CAP_DSHOW` on Windows?* It opens much faster than the default backend.

**Face detection**
10. *How does HOG detection work?* Histogram of Oriented Gradients describes local edge directions; a sliding window is scored by a linear SVM; overlapping detections are merged.
11. *HOG vs CNN/DNN detectors?* HOG is fast on CPU but weaker for profile faces, occlusion, and small faces; CNN detectors are more robust but heavier.
12. *Detection vs recognition?* Detection finds *where* a face is; recognition decides *who* it is.

**Embeddings / ML**
13. *What is an embedding?* A fixed-length numeric vector produced by a neural network such that distance reflects identity similarity.
14. *Did you train the model?* No. A pretrained dlib ResNet-style model is used for inference; my contribution is the pipeline, matching rule, quality checks, storage and attendance logic.
15. *Why several samples per person?* They capture pose/lighting variation; matching takes the nearest sample, which lowers false rejects.
16. *Does the model do face alignment?* dlib finds facial landmarks and warps the face to a canonical pose before the CNN.
17. *How would you improve accuracy?* Better cameras/lighting, more varied samples, per-site threshold calibration, newer models (e.g. ArcFace-style), averaging/centroid + k-NN with margin.
18. *Why not a classifier (SVM/softmax) over identities?* It needs retraining whenever a person is added; nearest-neighbour on embeddings supports adding/removing people instantly (open-set recognition).

**Similarity metrics and threshold**
19. *Euclidean vs cosine?* Euclidean measures straight-line distance; cosine measures angle. Dlib embeddings are tuned for Euclidean (typical same-person distance < ~0.6), so it is used; cosine is included for comparison.
20. *How do you choose the threshold?* Collect genuine pairs (same person) and impostor pairs, plot their distance distributions, choose the point that meets the required false-accept/false-reject trade-off (e.g. ROC/EER). The default 0.5 is conservative; I have **not** measured it on my own data.
21. *What are false accept and false reject?* Wrongly identifying a stranger as a user vs failing to recognise a registered user. Lowering the threshold trades one for the other.
22. *Is the confidence a probability?* No — it is a display heuristic from distance; calibrating a real probability needs labelled data.
23. *Why is "Unknown" important?* It makes this an open-set problem: a closest match always exists, so without a threshold every stranger would be assigned to someone.

**SQLite / databases**
24. *Why SQLite?* Zero-config, single file, ACID, enough for a single-machine desktop app.
25. *How are embeddings stored?* As a `float32` BLOB (128×4 = 512 bytes) validated before insert; read back with `np.frombuffer`.
26. *Why a transaction at registration?* User + all embeddings are saved atomically; an interruption leaves no half-registered user.
27. *Foreign keys / cascade?* `ON DELETE CASCADE` removes embeddings and attendance with the user; SQLite needs `PRAGMA foreign_keys=ON` per connection (done in `_connect`).
28. *How do you prevent SQL injection?* Parameterised queries everywhere; LIKE wildcards are escaped.
29. *Why one connection per operation?* Safe across the GUI and recognition threads (SQLite connections are not shared between threads by default).

**GUI**
30. *Why not update widgets from the worker thread?* Tkinter is not thread-safe; the worker stores the latest frame and the main thread polls with `after()`.
31. *How do you avoid conflicts over the webcam?* The dashboard stops recognition before opening registration; each component releases the camera in `stop/close`.

**Testing**
32. *How do you test without a webcam or model?* Fakes via injected backends, synthetic embeddings, fake clock, fake capture; plus a real-model smoke run on still photos.
33. *What would you test next?* A labelled dataset evaluation (ROC), GUI tests with a virtual display, a long-run camera soak test.

**Security / privacy**
34. *How is biometric privacy handled?* Local processing only, no images stored/logged, embeddings only, admin deletion of face data, git-ignored DB; recommend disk encryption and consent.
35. *Biggest security weakness?* No liveness detection (photo/screen spoofing) and unencrypted embeddings at rest.
36. *Can an embedding be turned back into a face?* Not trivially, but embeddings are still personal biometric data and can be used to identify someone, so they are protected like images.

**Architecture**
37. *Why layers (gui → services → core → database)?* Business logic is testable without Tk, replaceable parts (swap the encoder) and no circular imports.
38. *What are the performance trade-offs?* Processing every Nth frame and downscaling improve FPS but can miss brief appearances and lower small-face accuracy.
39. *How would you scale to many users?* Replace the linear scan with an ANN index (FAISS) and a server-side database; for hundreds of people the NumPy scan is already fast.
40. *What would you change for production?* Liveness, encrypted storage, authentication for admins, audit logs, retention policy, bias evaluation.

## D. Resume material
**Bullets**
- Built a modular desktop face-recognition attendance system in Python (OpenCV, dlib embeddings, SQLite, Tkinter) with real-time recognition on a background thread, configurable thresholds, and duplicate-attendance cooldown.
- Designed a layered architecture (GUI / services / core / database) with dependency injection, enabling 76 automated pytest tests that run without a webcam, covering matching, attendance rules, registration validation, and failure handling.
- Implemented privacy-conscious biometric handling: local-only processing, embedding-only storage, atomic registration transactions, per-user data deletion, and logging that excludes biometric data.

**Description (2–3 lines):** A Python desktop application that registers people from a webcam, recognises them in real time with pretrained 128-D face embeddings and a threshold-based matcher, and records de-duplicated attendance in SQLite with search and CSV export. Built with a testable, layered architecture and documented limitations and privacy considerations.

**Technologies:** Python, OpenCV, dlib / face_recognition, NumPy, Pandas, Pillow, Tkinter, SQLite, pytest, logging.

**ATS keywords:** Computer Vision, Face Recognition, Face Embeddings, OpenCV, Machine Learning, Python, SQLite, Unit Testing (pytest), Real-time Processing, Software Architecture, Biometric Privacy.

*(Add performance numbers to resume bullets only after you measure them, e.g. FPS and false-accept/false-reject on your own test set.)*
