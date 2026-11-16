@echo off
cd /d "%~dp0"
pip install -q -r requirements.txt -r webapp\requirements-web.txt
python webapp\app.py
