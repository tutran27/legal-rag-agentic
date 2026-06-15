$ErrorActionPreference = "Continue"

$repo = "D:\legal-agent-rag"
$checkpoint = Join-Path $repo "data\embedding_shards_harrier\qdrant_checkpoint.json"
$logFile = Join-Path $repo "data\ingest_shutdown_monitor.log"
$collection = "legal_agent_rag_harrier_idf"

function Write-MonitorLog([string]$message) {
    $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $message"
    Add-Content -LiteralPath $logFile -Value $line
}

function Shutdown-Computer([string]$reason) {
    Write-MonitorLog "SHUTDOWN: $reason"
    & shutdown.exe /s /t 60 /c "Legal RAG ingest monitor: $reason"
    exit
}

Write-MonitorLog "Monitor started."

while ($true) {
    $shard = 0
    if (Test-Path -LiteralPath $checkpoint) {
        try {
            $state = Get-Content -LiteralPath $checkpoint -Raw | ConvertFrom-Json
            $shard = [int]$state.shard
        } catch {
            Write-MonitorLog "Cannot read checkpoint: $($_.Exception.Message)"
        }
    }

    if ($shard -ge 101) {
        Shutdown-Computer "Ingest completed at shard $shard/101."
    }

    $processOutput = & wsl.exe bash -lc "ps -eo args= | grep '[p]ython -m scripts.modal_shards_to_qdrant' || true" 2>&1
    $processAlive = ($LASTEXITCODE -eq 0) -and ($processOutput -match "modal_shards_to_qdrant")

    if ($processAlive) {
        Write-MonitorLog "Healthy: shard $shard/101."
    } else {
        Shutdown-Computer "Ingest process stopped at shard $shard/101."
    }

    Start-Sleep -Seconds 60
}
