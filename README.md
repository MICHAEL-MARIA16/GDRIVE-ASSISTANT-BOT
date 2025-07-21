# 🧠 IntelliRAG AI Bot Assistant

🚀 **Live Demo:** [https://intellirag-bot.onrender.com](https://intellirag-bot.onrender.com)

IntelliRAG is an intelligent chatbot powered by Retrieval-Augmented Generation (RAG), designed to provide seamless document-based question-answering experiences. Built with LangChain, Google Gemini, and Qdrant, it offers live file uploads, background syncing, and a robust Flask/Streamlit interface.

## 🌐 Deployment

IntelliRAG is designed for flexible deployment, leveraging modern cloud infrastructure:

- **Backend:** Render (Flask + Docker)
- **Frontend:** Optional static HTML/JS UI (intellirag.html)
- **Vector Store:** Qdrant Cloud
- **Embeddings:** Google Generative AI Embeddings
- **LLM:** Gemini Pro / Fallback to OpenAI (Optional)

## 🔧 Features

IntelliRAG comes packed with features to ensure a smooth and efficient knowledge base interaction:

- ✅ **Real-time Knowledge Base Sync:** Automatically syncs from a local folder
- ✅ **Extensive File Support:** Handles PDF, DOCX, TXT, CSV, JSON, and various code files
- ✅ **Automatic Processing:** Effortless embedding and chunking of documents
- ✅ **Cloud-Powered Vector DB:** Utilizes Qdrant Cloud for high-performance vector search
- ✅ **Efficient Syncing:** Background sync thread with CLI control for knowledge base updates
- ✅ **Live API Endpoints:** Provides chat, reload, and stats endpoints for dynamic interaction
- ✅ **Flexible LLM Switching:** Ready for fallback to alternative LLMs if needed
- ✅ **Containerized Deployment:** Dockerized for easy and consistent deployment across environments

## 📁 File Structure

The project is organized for clarity and maintainability:

```
local-folder-chatbot/
├── chatbot.py              # Core RAG logic and sync pipeline
├── chatbot_api.py          # Flask application with RESTful endpoints
├── Dockerfile              # Production Docker container definition
├── docker-compose.yml      # Local Docker orchestration
├── requirements.txt        # Python dependencies
├── sync.py                 # Background syncing and embedding logic
├── intellirag.html         # Minimal frontend user interface
├── file_index.db           # SQLite index for knowledge base files
├── logs/                   # Sync and system logs
└── .env                    # Environment variables
```

## 🧪 API Endpoints

Interact with the IntelliRAG backend using these RESTful API endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Checks the health status of the API |
| `/chat` | POST | Submits a question to the chatbot |
| `/reload` | POST | Reloads the knowledge base and vector index |
| `/knowledge-base/files` | GET | Lists all indexed documents in the knowledge base |
| `/knowledge-base/statistics` | GET | Displays statistics about the knowledge base |

## ⚙️ Environment Variables

Configure your project by setting the following environment variables in a `.env` file:

```env
QDRANT_URL=https://YOUR-QDRANT-URL
QDRANT_API_KEY=your_qdrant_key
GOOGLE_API_KEY=your_gemini_api_key
```

## 🐳 Docker Usage

Easily set up and deploy IntelliRAG using Docker:

### Local Testing

```bash
docker-compose up --build
```

### Fly.io Deployment (Optional)

```bash
fly launch
fly deploy
```

## 🛠️ Technologies Used

IntelliRAG leverages a powerful stack of modern technologies:

- **Flask** – For building the robust REST API
- **LangChain** – Essential for RAG implementation and chain management
- **Qdrant** – As the high-performance vector database
- **Gemini Pro** – Google's cutting-edge Large Language Model
- **FAISS** – Optional fallback for vector indexing
- **rclone** – For optional Google Drive synchronization
- **Docker** – For efficient containerization and deployment
- **Fly.io / Render** – For seamless cloud deployment

## 💡 TODO

We have exciting plans for future enhancements:

- [ ] Add user authentication for secure access
- [ ] Implement WebSockets for live updates and real-time interaction
- [ ] Extend support for image and document previews
- [ ] Develop an analytics dashboard for performance monitoring

## 📜 License

This project is open-source and released under the MIT License.

## 🙌 Acknowledgements

Special thanks to the teams behind LangChain, Qdrant, and Google AI for providing the foundational tools that made this project possible.

## 💬 Contact

For feedback, questions, or collaboration opportunities, please reach out to [mariaselciya.m@gmail.com](mailto:mariaselciya.m@gmail.com)