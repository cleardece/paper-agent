@echo off
cd /d D:\paper-agent
.venv\Scripts\python.exe -m uvicorn web.app:app --host 0.0.0.0 --port 8000 --log-level info
pause
