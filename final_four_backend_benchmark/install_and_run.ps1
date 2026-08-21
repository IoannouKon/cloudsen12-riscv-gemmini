$ErrorActionPreference='Stop';$p='C:\Users\User\cloudsen12_pure_c';$py='C:\Users\User\cloudsen12-qat-env\Scripts\python.exe';$h=Split-Path -Parent $MyInvocation.MyCommand.Path
Copy-Item "$h\run_windows_pytorch.py" "$p\benchmark\run_windows_pytorch_final.py" -Force
Copy-Item "$h\run_wsl_backends.sh" "$p\benchmark\run_wsl_backends_final.sh" -Force
Set-Location $p
& $py -u .\benchmark\run_windows_pytorch_final.py
wsl bash -lc 'cd /mnt/c/Users/User/cloudsen12_pure_c && bash benchmark/run_wsl_backends_final.sh'
Get-Content .\benchmark\results\final_four_backend\final_summary.json
