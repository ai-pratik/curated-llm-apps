# 🐙 GitHub MCP Agent

A Streamlit app that lets you explore and analyze GitHub repositories using **natural language** — powered by local LLMs via [Ollama](https://ollama.com), the [Agno](https://github.com/agno-agi/agno) agent framework, and GitHub's official [MCP Server](https://github.com/github/github-mcp-server).

> **No cloud API keys needed** — runs entirely on your local machine with Ollama.

---

## ✨ Features

- **100% Local LLMs** — Uses Ollama models (llama3.2, gemma3, etc.) — no OpenAI or cloud APIs required
- **Natural Language Queries** — Ask about issues, PRs, repo activity in plain English
- **MCP Integration** — Connects to GitHub's official MCP server for real-time API access
- **Interactive Streamlit UI** — Select models, switch query types, and get markdown-formatted results
- **Docker-based MCP Server** — GitHub MCP server runs in an isolated Docker container

## 🏗️ Architecture

```
┌─────────────────────────────────────────────┐
│              Streamlit UI                   │
│         (github_agent.py)                   │
├─────────────────────────────────────────────┤
│        Agno Agent Framework                 │
│   ┌─────────────┐  ┌─────────────────────┐ │
│   │ Ollama LLM  │  │    MCPTools         │ │
│   │ (local)     │  │  (tool interface)   │ │
│   └─────────────┘  └────────┬────────────┘ │
│                              │              │
│                    ┌─────────▼────────────┐ │
│                    │  GitHub MCP Server   │ │
│                    │  (Docker container)  │ │
│                    └─────────┬────────────┘ │
│                              │              │
│                    ┌─────────▼────────────┐ │
│                    │    GitHub REST API   │ │
│                    └─────────────────────┘ │
└─────────────────────────────────────────────┘
```

## 📋 Prerequisites

| Requirement | Purpose |
|---|---|
| **Python 3.8+** | Runtime |
| **[Ollama](https://ollama.com)** | Local LLM inference |
| **[Docker](https://www.docker.com/get-started)** | Runs the GitHub MCP server |
| **GitHub Personal Access Token** | API authentication ([create one here](https://github.com/settings/tokens) with `repo` scope) |

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/ai-pratik/curated-llm-apps.git
cd curated-llm-apps/mcp-agent
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Pull an Ollama model

```bash
ollama pull llama3.2:latest
```

### 4. Make sure Docker is running

```bash
docker --version
docker ps
```

### 5. Launch the app

```bash
streamlit run github_agent.py
```

### 6. In the Streamlit UI

1. Select your **Ollama model** from the sidebar (llama3.2 recommended for tool calling)
2. Enter your **GitHub Personal Access Token** in the sidebar
3. Specify a repository to analyze (e.g. `ai-pratik/curated-llm-apps`)
4. Choose a query type or write a custom query
5. Click **🚀 Run Query**

## 💻 How It Works

The core agent is built in ~50 lines using **Agno** and **MCP**:

```python
from agno.agent import Agent
from agno.models.ollama import Ollama
from agno.tools.mcp import MCPTools
from mcp import StdioServerParameters

# Connect to GitHub's official MCP server via Docker
server_params = StdioServerParameters(
    command="docker",
    args=["run", "-i", "--rm",
          "-e", "GITHUB_PERSONAL_ACCESS_TOKEN",
          "-e", "GITHUB_TOOLSETS",
          "ghcr.io/github/github-mcp-server"],
    env={
        "GITHUB_PERSONAL_ACCESS_TOKEN": os.getenv("GITHUB_TOKEN"),
        "GITHUB_TOOLSETS": "repos,issues,pull_requests"
    }
)

async with MCPTools(server_params=server_params) as mcp_tools:
    agent = Agent(
        model=Ollama(id="llama3.2:latest"),
        tools=[mcp_tools],
        instructions="You are a GitHub assistant...",
        markdown=True,
    )
    response = await agent.arun("Find open issues in owner/repo")
```

The MCP server exposes GitHub API operations as tools that the LLM can invoke autonomously — the agent decides which API calls to make based on your natural language query.

## 📝 Example Queries

#### Issues
- "Show me issues by label"
- "What issues are being actively discussed?"
- "Find issues labeled as bugs"

#### Pull Requests
- "What PRs need review?"
- "Show me recent merged PRs"
- "Find PRs with conflicts"

#### Repository
- "Show repository health metrics"
- "Show repository activity patterns"
- "Analyze code quality trends"

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| **Agent Framework** | [Agno](https://github.com/agno-agi/agno) |
| **LLM** | [Ollama](https://ollama.com) (local) |
| **Tool Protocol** | [Model Context Protocol (MCP)](https://modelcontextprotocol.io) |
| **MCP Server** | [github/github-mcp-server](https://github.com/github/github-mcp-server) (Docker) |
| **UI** | [Streamlit](https://streamlit.io) |

## 📜 License

MIT
