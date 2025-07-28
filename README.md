# 📄 PDF Outline Extractor (Dockerized)

This project extracts outlines/headings from PDF files and generates a structured JSON file using `sentence-transformers`, `pdfplumber`, and font-size heuristics.

> ✅ Fully Dockerized | ✅ Offline Capable | ✅ Model Cached Locally

---

## 📆 Folder Structure

```
├── app/
│   ├── main.py              # Main PDF extraction script
│   ├── input/               # Place your input PDFs here
│   │   └── test.pdf
│   ├── output/              # JSON output will be saved here
│   └── localmodel/          # Pre-downloaded sentence-transformers model
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## 🚀 How to Run

### 1. 🐳 Build the Docker Image

```bash
docker build -t pdf-outline-extractor:latest .
```

---

### 2. 📂 Prepare Input & Model

* Place your PDF(s) inside:

  ```
  app/input/
  ```

* Ensure `app/localmodel/` contains the pre-downloaded `all-MiniLM-L6-v2` model (see below).

---

### 3. 🧠 Download Model (if not already present)

Use this Python script once to download and save the model:

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
model.save("app/localmodel")
```

Run it from the project root:

```bash
python app/download_model.py
```

---

### 4. 🏃‍♂️ Run the Container

```bash
docker run --rm \
  -v $(pwd)/app/input:/app/input \
  -v $(pwd)/app/output:/app/output \
  -v $(pwd)/app/localmodel:/app/localmodel \
  pdf-outline-extractor:latest
```

---

## 📤 Output

For an input `test.pdf`, you'll get:

```bash
app/output/test.json
```

Containing structured heading info:

```json
[
  {
    "level": "H1",
    "text": "Introduction",
    "page": 1
  },
  ...
]
```

---

## 🛠️ Tech Stack

* 🐍 Python 3.10
* 🧠 sentence-transformers
* 📄 pdfplumber
* 🐳 Docker

---

## 📌 Notes

* Works offline using local model (`localmodel/`)
* If running in Codespaces or a cloud IDE, use absolute paths in `docker run`
* Output file is saved with the same name as input PDF but with a `.json` extension

---
