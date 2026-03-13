@echo off
cd %~dp0
.\venv\Scripts\python.exe backend\ingestion.py --year %1 --track %2 --session %3 --driver %4
