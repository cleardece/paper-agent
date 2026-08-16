param(
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"

docker compose up -d
& $Python -m uvicorn web.app:app --host 0.0.0.0 --port 8000 --log-level info
