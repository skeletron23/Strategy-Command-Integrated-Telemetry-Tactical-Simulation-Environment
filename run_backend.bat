@echo off
cd %~dp0backend
..\venv\Scripts\uvicorn.exe main:app --reload --port 8000
