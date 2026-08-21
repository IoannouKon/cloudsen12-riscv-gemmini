$ErrorActionPreference="Stop"
$project="C:\Users\User\cloudsen12_pure_c";$here=Split-Path -Parent $MyInvocation.MyCommand.Path
Copy-Item $here "$project\pure_c_int8" -Recurse -Force
Write-Host "Installed pure_c_int8. In WSL run:"
Write-Host "cd /mnt/c/Users/User/cloudsen12_pure_c && sudo apt install -y python3-numpy && bash pure_c_int8/build_and_validate_wsl.sh"
