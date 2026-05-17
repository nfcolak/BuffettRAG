# UZH CL — RAG Research Project Workspace

## BuffettRAG Frontend

The primary BuffettRAG frontend is now the React/Vite app in `frontend/`.

```bash
cd frontend
npm install
VITE_BACKEND_URL=http://localhost:8000 npm run dev
```

Then open `http://127.0.0.1:5173`.

The old Streamlit frontend (`app.py`) is kept as a fallback/legacy UI.

Welcome to your team workspace on [Nuvolos](https://nuvolos.cloud). This document explains the tools available to you and how they work together. For more details on the platform, refer to the [Nuvolos documentation](https://docs.nuvolos.com) or contact Nuvolos support for assistance. 

***

## Your Applications

Each team member has their own instance with <strong>four applications</strong>. The **Master instance** additionally includes a **Trainer** app.

| App | What it is | What it's for |
| --- | ---------- | ------------- |
| **Editor** | VS Code (1 NCU) | General-purpose code editing, scripting, and file management |
| **Frontend** | VS Code (2 NCU) | Building and serving your RAG frontend (e.g., Streamlit, Gradio) |
| **Backend** | VS Code (2 NCU) | Running your RAG backend or API server |
| **Database** | pgvector (2 NCU) | PostgreSQL database with vector search support for embeddings |
| **Trainer** | PyTorch (Master only) | Fine-tuning and training models — available only on the Master instance |

> **NCU** = Nuvolos Compute Unit. Apps with more NCUs have access to more CPU and memory. 1 NCU = 1 vCPU with 4 GB RAM. [web:3]

***

## Networking Between Apps

The <strong>Frontend</strong>, <strong>Backend</strong>, and **Database** apps are configured with **instance-wide networking** (`ipc_mode: instance`). This means:

* All three apps within the same instance can reach each other over fixed hostnames. The hostname is shown in the Nuvolos UI under Applications → “… > CONFIGURE”.
* The **Backend** can connect directly to the **Database** (PostgreSQL on port 5432).
* The **Frontend** can communicate with the **Backend API** without external routing.

The **Editor** app runs in isolation and does **not** share the network namespace with the other apps. 

**Typical connection flow:**

```text
Frontend → Backend → Database (pgvector)
   ↑          ↑          ↑
   └── all reachable via unique, fixed hostnames within the same instance ──┘
```

See the \[Nuvolos documentation\]\(https://docs\.nuvolos\.com/features/applications/configuring\-applications\#connecting\-to\-apps\-from\-other\-applications\) for more\.

***

## Large File Storage (LFS)

Each team space includes a <strong>shared 30 GiB file storage</strong>, mounted at:

```text
/space_mounts/pars
```

* The mount is **read/write** and shared across all instances in your space.
* Use it for model weights, checkpoints, datasets, or other large shared files.
* Changes made by one team member are visible to all team members.

***

## Team Credits

Your team has a **shared credit pool** that is consumed while apps are running. To conserve credits:

* **Stop apps** when not in use.
* Use the **Editor** (1 NCU) for lightweight tasks.
* Check remaining credits in your Nuvolos dashboard under **Space Settings**
* You can use `git` with SSH key support for private repositories. We recommend each instance work on its own branch, with the Master using the `main` branch.

***

## Scaling the Trainer App

You can scale the **Trainer** app for larger training workloads:

1. Go to the <strong>Master instance</strong>.
2. Open the **Trainer** app settings.
3. Increase NCU allocation for more CPU/memory (or request GPU access if available).
4. Start your training job.
5. **Scale back down** once training is complete to save credits. 

***

## Quick Start

1. **Accept your invitation** from the email you received.
2. **Open your instance** — it’s listed under your group’s space. 
3. **Start the Editor** to explore files and set up your project.
4. **Start the Database** and initialize your pgvector schema.
5. **Start Backend and Frontend** to develop your RAG pipeline.
6. Use `/space_mounts/pars` for shared model weights and large files.

***

## Questions?

For any questions, please contact: [stylianos.psychias@uzh.ch](mailto:stylianos.psychias@uzh.ch)
