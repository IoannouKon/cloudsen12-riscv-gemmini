$ErrorActionPreference="Stop"
$p="C:\Users\User\cloudsen12_pure_c"
$h=Split-Path -Parent $MyInvocation.MyCommand.Path
Copy-Item "$h\fix_and_run_wsl.sh" "$p\pure_c_int8\fix_float_align_generator_wsl.sh" -Force
wsl bash -lc "cd /mnt/c/Users/User/cloudsen12_pure_c && bash pure_c_int8/fix_float_align_generator_wsl.sh"
