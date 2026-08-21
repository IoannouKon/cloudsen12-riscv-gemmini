$ErrorActionPreference="Stop";$p="C:\Users\User\cloudsen12_pure_c";$py="C:\Users\User\cloudsen12-qat-env\Scripts\python.exe";$h=Split-Path -Parent $MyInvocation.MyCommand.Path
Copy-Item "$h\run_strict_retry.py" "$p\run_strict_retry.py" -Force;Set-Location $p;& $py -m py_compile .\run_strict_retry.py;& $py -u .\run_strict_retry.py;Get-Content .\qat\deployment\v9\q31_retry_strict\result.json
