# Getting started

Install the project using README.md, then run the offline demo or the real-data workflow. Environments, downloaded datasets, trained models and reports are generated locally; they are not committed to this repository.

1. Read **README.md** for commands and design.
2. After the real-data workflow, open **runs/uci/report.html** for the validation report.
3. After the demo below, open **runs/my-demo/monitoring/report.html** for the synthetic drift demonstration.
4. Read **docs/RESULTS.md** for measured metrics and supported resume bullets.
5. Read **docs/INTERVIEW_GUIDE.md** and inspect the source so you can explain every decision.

From PowerShell inside this project directory:

```powershell
.\.venv\Scripts\creditrisk.exe --help
.\.venv\Scripts\creditrisk.exe demo --out runs/my-demo
.\.venv\Scripts\python.exe -m pytest -q
```

Use a fresh directory when running a new experiment. No API keys, paid services, database server, or web server are required. SQLite is built into Python.
