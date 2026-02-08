# Duplicate Document scanner  🛡️

**Intelligent Semantic Plagiarism & Duplicate Detector**

NexusCheck Ultimate is a next-generation plagiarism detection tool that goes beyond traditional keyword matching. Powered by advanced AI (Transformer models), it detects **semantic similarity**, meaning it can catch plagiarism even if the words are paraphrased, rearranged, or translated from another language.

## 🚀 Key Features

* **🧠 Semantic Analysis:** Uses Vector Embeddings to understand the *meaning* of text, not just the spelling.
* **🌐 Multi-Language Support:** Supports 50+ languages (including English, Hindi, Spanish, French) using the `paraphrase-multilingual-MiniLM-L12-v2` model.
* **📷 OCR Capability:** Reads text from **Images (PNG, JPG)** and scanned PDFs using Tesseract OCR.
* **🌍 Live Web Search:** Scrapes the live internet (via DuckDuckGo) to find plagiarized content from external sources.
* **📊 Visual Dashboard:** Features a modern, glassmorphism UI with real-time Pie Charts and Bar Graphs for risk analysis.
* **🛡️ Privacy Focused:** Runs entirely locally on your machine. No documents are stored on external clouds.
* **⚡ Hybrid Search:** Uses a fail-safe mechanism to ensure search results are returned even if one method is blocked.

## 🛠️ Tech Stack

* **Backend:** Python, FastAPI
* **AI Engine:** Sentence-Transformers (Hugging Face)
* **Computer Vision:** Tesseract OCR, Pillow
* **Search Engine:** DuckDuckGo Search (DDGS), Requests, BeautifulSoup
* **Frontend:** HTML5, Tailwind CSS, Chart.js

## ⚙️ Installation & Setup

Follow these steps to run the project locally.

### 1. Prerequisites
* Python 3.8+
* [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) installed on your system.

