Great choice with the deployment! Here's a polished and professional `README.md` file tailored for your IntelliRAG AI bot assistant project:

---

```markdown
# 🧠 IntelliRAG AI Bot Assistant

![Status](https://img.shields.io/badge/status-deployed-brightgreen)  
🚀 Live Demo: [https://intellirag-bot.onrender.com](https://intellirag-bot.onrender.com)

IntelliRAG is an intelligent chatbot powered by Retrieval-Augmented Generation (RAG), built with LangChain, Google Gemini, and Qdrant. It supports live file uploads, background syncing, and a rich Flask/Streamlit interface for seamless document-based QA experiences.

---

## 🌐 Deployment

- **Backend**: Render (Flask + Docker)
- **Frontend**: Optional static HTML/JS UI (`intellirag.html`)
- **Vector Store**: Qdrant Cloud
- **Embeddings**: Google Generative AI Embeddings
- **LLM**: Gemini Pro / Fallback to OpenAI (Optional)

---

## 🔧 Features

- ✅ Real-time knowledge base sync from local folder
- ✅ Supports PDF, DOCX, TXT, CSV, JSON, and code files
- ✅ Automatic embedding + chunking
- ✅ Qdrant cloud vector DB
- ✅ Background sync thread + CLI control
- ✅ Live API endpoints (chat, reload, stats)
- ✅ Fallback-ready LLM switching
- ✅ Dockerized for easy deployment

---

## 📁 File Structure

```

local-folder-chatbot/
├── chatbot.py              # Core RAG logic and sync pipeline
├── chatbot\_api.py          # Flask app with all RESTful endpoints
├── Dockerfile              # Production container
├── docker-compose.yml      # Local Docker orchestration
├── requirements.txt        # Python dependencies
├── sync.py                 # Background syncing + embedding logic
├── intellirag.html         # Minimal frontend UI
├── file\_index.db           # SQLite index for KB files
├── logs/                   # Sync and system logs
└── .env                    # Environment variables

```

---

## 🧪 API Endpoints

| Endpoint                         | Description                         |
|----------------------------------|-------------------------------------|
| `GET /health`                   | Health check                        |
| `POST /chat`                    | Ask a question                      |
| `POST /reload`                  | Reload KB and vector index          |
| `GET /knowledge-base/files`     | List all indexed documents          |
| `GET /knowledge-base/statistics`| Show KB stats                       |

---

## ⚙️ Environment Variables

```

QDRANT\_URL=[https://YOUR-QDRANT-URL](https://YOUR-QDRANT-URL)
QDRANT\_API\_KEY=your\_qdrant\_key
GOOGLE\_API\_KEY=your\_gemini\_api\_key

````

---

## 🐳 Docker Usage

### Local Testing
```bash
docker-compose up --build
````

### Fly.io Deployment (Optional)

```bash
fly launch
fly deploy
```

---

## 🛠️ Technologies Used

* **Flask** – REST API
* **LangChain** – RAG + Chain Management
* **Qdrant** – Vector DB
* **Gemini Pro** – Google LLM
* **FAISS** – Optional fallback
* **rclone** – Google Drive sync (optional)
* **Docker** – Containerization
* **Fly.io / Render** – Deployment

---

## 💡 TODO

* [ ] Add user authentication
* [ ] Add WebSocket for live updates
* [ ] Extend support for image/doc previews
* [ ] Add analytics dashboard

---

## 📜 License

MIT License — open source with ❤️

---

## 🙌 Acknowledgements

Special thanks to LangChain, Qdrant, and Google AI for the tools that made this project possible.

---

## 💬 Contact

For feedback, questions, or collaborations: [maria.selciya@example.com](mailto:maria.selciya@example.com)

```

