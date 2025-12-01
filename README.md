## Contactless Authentication with Liveness Detection

#### CS228 Project

---

### 📌 Problem Statement

With the advancement of technology in the biometric field, healthcare facilities are adopting contactless authentication methods to improve security, efficiency, and patient safety. While contactless check-in systems aren’t necessary for every clinic, they can provide additional value in busy clinics where hygiene, efficiency, and accessibility are crucial.

However, current facial authentication systems are vulnerable to spoofing attacks. Our project aims to develop a secure facial recognition system that verifies user identity while maintaining privacy.

### 🎯 Objectives

- Implement an enrollment module that builds a face embedding template per user.
- Develop an identity verification module using cosine similarity between live embeddings and stored templates.
- Integrate a liveness detection (PAD) system to detect spoofing via both action-based and learned cues.
- Combine both identity and liveness scores for a secure authentication decision.

### Datasets

| Dataset  | Purpose            | Description                                                                        |
| -------- | ------------------ | ---------------------------------------------------------------------------------- |
| VGGFace2 | Identity Check     | 3 million images, 9000+ individuals (10+ images per person) with lots of diversity |
| Face Anti-Spoofing Dataset from NUAA   | Liveness Detection |  Over 4,000 images including real person and fake person data
|
