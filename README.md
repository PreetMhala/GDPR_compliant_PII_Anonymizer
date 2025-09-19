# AI Enabler: Anonymizer for European NATCOs

<p align="center">
  <img src="https://img.shields.io/badge/Status-Completed-green.svg" alt="Project Status">
  <img src="https://img.shields.io/badge/Language%20Support-Multilingual-blue.svg" alt="Language Support">
  <img src="https://img.shields.io/badge/Compliance-GDPR-red.svg" alt="GDPR Compliance">
  <img src="https://img.shields.io/badge/Frameworks-PyTorch%20%7C%20SpaCy%20%7C%20Presidio-orange.svg" alt="Frameworks">
</p>

This repository contains the source code and documentation for the **AI Enabler: Anonymizer for European NATCOs** project, a robust, multilingual PII anonymization pipeline designed for real-time chatbot integration at Deutsche Telekom. This project was completed as a requirement for the degree of **Bachelor of Technology in Computer Science and Engineering** by **Prit Mhala** (MIS No: 112115089) at the **Indian Institute of Information Technology, Pune**.

---

## 🧐 Abstract

The widespread use of chatbots globally has led to a massive collection of user-generated data, often containing **Personally Identifiable Information (PII)**. To comply with the **General Data Protection Regulation (GDPR)**, this sensitive data must be anonymized before being logged or used for downstream tasks like model training and analytics.

This project presents a scalable, multilingual PII anonymization solution for seven European NATCOs: Germany, Hungary, Poland, Greece, Croatia, Montenegro, and the UK. The pipeline uses an ensemble approach, combining **Named Entity Recognition (NER)** models (from SpaCy, HuggingFace, and Flair), language-specific regular expressions, and the Presidio framework. This hybrid method achieves over **90% value-level accuracy** in detecting and masking a wide range of PII entities while preserving the semantic context of user inputs.

The system is designed for low-latency, real-time integration, offering a cost-effective and extensible alternative to large-scale LLM-based solutions.

---

## 💡 Key Features

* **Multilingual Support**: Accurately anonymizes PII in seven European languages: English, German, Hungarian, Polish, Greek, Croatian, and Montenegrin.
* **Hybrid Anonymization Pipeline**: Combines rule-based methods (Regex), state-of-the-art NER models (SpaCy, Flair, HuggingFace), and the Presidio framework for high accuracy and robust detection.
* **GDPR Compliant**: Replaces sensitive data with type-specific placeholder tags (e.g., `[PHONE_NUMBER]`, `[EMAIL]`) instead of a generic `[PII]` tag to maintain data utility for analytics while ensuring privacy.
* **Low Latency & Scalable**: Optimized for high-throughput, real-time chatbot environments, making it suitable for processing millions of queries per day.
* **Modular Architecture**: Each language has a dedicated configuration file, allowing for easy expansion to new languages or PII entity types.

---

## 🔧 Workflow Diagram

The anonymization pipeline follows a structured, multi-step process to ensure all PII is detected and masked accurately.

!(https://i.imgur.com/g6sU0nS.png)

1.  **Input**: A user query from a chatbot is received.
2.  **Language Identification**: The query language is identified.
3.  **Model Configuration**: The system loads the correct configuration file and the corresponding language-specific NER models and regex patterns.
4.  **PII Detection**: The query is processed by the ensemble of models (SpaCy, Flair, HuggingFace) and custom regex patterns to detect all PII entities.
5.  **Presidio Anonymization**: The Presidio framework takes the detected entities, resolves any overlaps, and replaces the sensitive information with the correct placeholder tags.
6.  **Output**: The anonymized query is returned for secure logging and storage.

---

