[CmdletBinding()]
param([string]$Python='python',[ValidateSet('A','D','both')][string]$Backend='both')
$ErrorActionPreference='Stop'
Set-Location -LiteralPath $PSScriptRoot
& $Python -c "import sys; assert sys.version_info[:2]==(3,12), 'Server requires Python 3.12'"
if($LASTEXITCODE -ne 0){throw 'Install Python 3.12 and pass -Python with its full path.'}
if(-not(Test-Path -LiteralPath '.venv/Scripts/python.exe')){
    & $Python -m venv .venv
    if($LASTEXITCODE -ne 0){throw 'Could not create virtual environment.'}
}
$runtime=Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
& $runtime -m pip install --index-url https://pypi.org/simple --upgrade pip
if($LASTEXITCODE -ne 0){throw 'pip setup failed.'}
& $runtime -m pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.7.1 torchvision==0.22.1
if($LASTEXITCODE -ne 0){throw 'CUDA PyTorch installation failed.'}
& $runtime -m pip install --index-url https://pypi.org/simple -r requirements-server.txt
if($LASTEXITCODE -ne 0){throw 'Python dependencies failed.'}
& $runtime -B prepare_assets.py --backend $Backend
if($LASTEXITCODE -ne 0){throw 'Model dependency preparation failed.'}
$doctorArgs=@('-B','-m','sam_ad.doctor')
if($Backend -ne 'both'){$doctorArgs+=@('--backend',$Backend)}
& $runtime @doctorArgs
if($LASTEXITCODE -ne 0){throw 'Doctor failed; resolve the reported dependency/CUDA/integrity issue.'}
Write-Output 'Ready. Run .\start_A.ps1 or .\start_D.ps1 in separate terminals.'
