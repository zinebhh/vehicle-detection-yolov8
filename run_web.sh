#!/usr/bin/env bash
# Lance le site : http://localhost:5000
cd "$(dirname "$0")" && pip install -q -r requirements.txt -r webapp/requirements-web.txt && python webapp/app.py
