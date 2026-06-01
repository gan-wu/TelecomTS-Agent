# GitHub Upload And QR Display

## 1. Initialize Git

```cmd
cd /d D:\WG\python_code\TelecomTS_QA_github
git init
git add .
git commit -m "Initial GitHub display package"
```

## 2. Create Remote Repository

Create a new empty repository on GitHub, for example:

```text
TelecomTS-Agent
```

Then connect and push:

```cmd
git remote add origin https://github.com/<your-user-name>/TelecomTS-Agent.git
git branch -M main
git push -u origin main
```

## 3. Generate QR Code

After the repository is public, copy the GitHub repository URL:

```text
https://github.com/<your-user-name>/TelecomTS-Agent
```

Use any QR code generator, or GitHub profile README tools, to generate a QR image from the URL. Put the QR code in the resume near the project name.

## 4. Interview Display Flow

Recommended explanation order:

1. Open README and show the architecture section.
2. Show `src/tools/tool_router.py` and `src/tools/telecom_tools.py` for Tool Calling.
3. Show `src/graph/telecom_graph.py` for LangGraph workflow.
4. Show `src/rag/hybrid_retriever.py` and `src/tools/telecom_knowledge_tool.py` for RAG.
5. Show `results/added_tech_effect_report.md` for experiment evidence.

