param(
    [ValidateSet('serve', 'verify')]
    [string]$Mode = 'serve',
    [string]$TeamRepo = '',
    [string]$ModelConfig = '',
    [string]$Credentials = ''
)

$ErrorActionPreference = 'Stop'
$localProject = Split-Path -Parent $PSScriptRoot
$launcherConfig = Join-Path $localProject 'outputs/local-stack/launcher.local.json'
$launcherSettings = $null
if (Test-Path -LiteralPath $launcherConfig) {
    $launcherSettings = Get-Content -LiteralPath $launcherConfig -Raw | ConvertFrom-Json
}
if (-not $TeamRepo) {
    $TeamRepo = if ($launcherSettings) { $launcherSettings.teamRepo } else { Split-Path -Parent $localProject }
}
if (-not $ModelConfig) {
    $ModelConfig = if ($launcherSettings) { $launcherSettings.modelConfig } else { Join-Path $localProject 'config.postgres.local.json' }
}
if (-not $Credentials) {
    $Credentials = if ($launcherSettings) { $launcherSettings.credentials } else { Join-Path $localProject 'outputs/postgres/runtime.local.json' }
}
$localPython = Join-Path $localProject '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $localPython)) {
    $localPython = (Get-Command python -ErrorAction Stop).Source
}
$launcherArguments = @(
    (Join-Path $PSScriptRoot 'local_stack_2024141460331.py'), $Mode,
    '--team-repo', $TeamRepo,
    '--model-config', $ModelConfig,
    '--credentials', $Credentials
)
& $localPython @launcherArguments
exit $LASTEXITCODE
