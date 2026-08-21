$ErrorActionPreference="Stop";$p="C:\Users\User\cloudsen12_pure_c";$h=Split-Path -Parent $MyInvocation.MyCommand.Path
Copy-Item "$h\apply_build_test_wsl.sh" "$p\pure_c_int8\apply_build_test_float_align_wsl.sh" -Force
wsl bash -lc "cd /mnt/c/Users/User/cloudsen12_pure_c && bash pure_c_int8/apply_build_test_float_align_wsl.sh"
