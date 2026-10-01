@echo off
REM Planificateur de tâches : signaux d'actualité des joueurs (Nimble, Ollama local), sans commit.
set PYTHONIOENCODING=utf-8
cd /d "%~dp0.."
python -W ignore "scripts\signaux_actualite.py" >> "automation\logs\signaux_actualite.log" 2>&1
