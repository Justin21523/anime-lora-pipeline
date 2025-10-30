# Windows Long-Running Setup Script
# Run this in PowerShell as Administrator

Write-Host "======================================================================" -ForegroundColor Green
Write-Host "Windows 長時間運行設置腳本" -ForegroundColor Green
Write-Host "======================================================================" -ForegroundColor Green
Write-Host ""
Write-Host "此腳本將配置 Windows 以支持長時間運行任務" -ForegroundColor Yellow
Write-Host ""

# Check if running as Administrator
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

if (-not $isAdmin) {
    Write-Host "❌ 錯誤：請以管理員身份運行此腳本" -ForegroundColor Red
    Write-Host ""
    Write-Host "右鍵點擊 PowerShell，選擇 '以系統管理員身分執行'" -ForegroundColor Yellow
    Write-Host ""
    Read-Host "按 Enter 鍵退出"
    exit 1
}

Write-Host "✅ 管理員權限確認" -ForegroundColor Green
Write-Host ""

# Confirm before proceeding
Write-Host "將要進行以下設置：" -ForegroundColor Cyan
Write-Host "  1. 禁用自動睡眠 (接電源時)" -ForegroundColor White
Write-Host "  2. 禁用自動休眠" -ForegroundColor White
Write-Host "  3. 禁用閒置自動登出" -ForegroundColor White
Write-Host "  4. 禁用螢幕保護程式" -ForegroundColor White
Write-Host ""

$confirm = Read-Host "是否繼續？(Y/N)"
if ($confirm -ne "Y" -and $confirm -ne "y") {
    Write-Host "已取消" -ForegroundColor Yellow
    exit 0
}

Write-Host ""
Write-Host "開始配置..." -ForegroundColor Green
Write-Host ""

# 1. Disable sleep and hibernate (when plugged in)
Write-Host "[1/4] 配置電源設置..." -ForegroundColor Cyan

try {
    # Monitor timeout (when plugged in)
    powercfg /change monitor-timeout-ac 0
    Write-Host "  ✅ 螢幕關閉時間：從不 (接電源)" -ForegroundColor Green

    # Standby timeout (when plugged in)
    powercfg /change standby-timeout-ac 0
    Write-Host "  ✅ 睡眠時間：從不 (接電源)" -ForegroundColor Green

    # Hibernate timeout (when plugged in)
    powercfg /change hibernate-timeout-ac 0
    Write-Host "  ✅ 休眠時間：從不 (接電源)" -ForegroundColor Green

    # Keep settings reasonable on battery
    powercfg /change monitor-timeout-dc 15
    powercfg /change standby-timeout-dc 30
    Write-Host "  ℹ️  電池模式保持合理設置 (15/30分鐘)" -ForegroundColor Gray
}
catch {
    Write-Host "  ❌ 電源設置失敗: $_" -ForegroundColor Red
}

Write-Host ""

# 2. Disable automatic logoff
Write-Host "[2/4] 禁用自動登出..." -ForegroundColor Cyan

try {
    # Disable inactivity timeout
    $result = reg add "HKLM\SOFTWARE\Microsoft\Windows\CurrentVersion\Policies\System" /v InactivityTimeoutSecs /t REG_DWORD /d 0 /f 2>&1

    if ($LASTEXITCODE -eq 0) {
        Write-Host "  ✅ 閒置自動登出：已禁用" -ForegroundColor Green
    } else {
        Write-Host "  ⚠️  可能需要手動設置群組原則" -ForegroundColor Yellow
    }
}
catch {
    Write-Host "  ❌ 設置失敗: $_" -ForegroundColor Red
}

Write-Host ""

# 3. Disable screensaver
Write-Host "[3/4] 禁用螢幕保護程式..." -ForegroundColor Cyan

try {
    # Disable screensaver
    $result = reg add "HKCU\Control Panel\Desktop" /v ScreenSaveActive /t REG_SZ /d 0 /f 2>&1

    if ($LASTEXITCODE -eq 0) {
        Write-Host "  ✅ 螢幕保護程式：已禁用" -ForegroundColor Green
    }
}
catch {
    Write-Host "  ❌ 設置失敗: $_" -ForegroundColor Red
}

Write-Host ""

# 4. Verify settings
Write-Host "[4/4] 驗證設置..." -ForegroundColor Cyan

try {
    Write-Host ""
    Write-Host "  當前電源方案設置 (接電源時)：" -ForegroundColor White

    # Get current power scheme
    $scheme = (powercfg /getactivescheme).Split()[3]

    # Check sleep settings
    $sleepAC = (powercfg /query $scheme SUB_SLEEP STANDBYIDLE | Select-String "Current AC Power Setting Index").ToString().Split()[-1]
    $hibernateAC = (powercfg /query $scheme SUB_SLEEP HIBERNATEIDLE | Select-String "Current AC Power Setting Index").ToString().Split()[-1]

    if ($sleepAC -eq "0x00000000") {
        Write-Host "    ✅ 睡眠：從不" -ForegroundColor Green
    } else {
        Write-Host "    ⚠️  睡眠：$([int]$sleepAC / 60) 分鐘" -ForegroundColor Yellow
    }

    if ($hibernateAC -eq "0x00000000") {
        Write-Host "    ✅ 休眠：從不" -ForegroundColor Green
    } else {
        Write-Host "    ⚠️  休眠：$([int]$hibernateAC / 60) 分鐘" -ForegroundColor Yellow
    }
}
catch {
    Write-Host "  ⚠️  無法完全驗證設置" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "======================================================================" -ForegroundColor Green
Write-Host "配置完成！" -ForegroundColor Green
Write-Host "======================================================================" -ForegroundColor Green
Write-Host ""

Write-Host "✅ 完成的設置：" -ForegroundColor Cyan
Write-Host "  • 電源管理：系統不會自動睡眠或休眠" -ForegroundColor White
Write-Host "  • 自動登出：已禁用閒置登出" -ForegroundColor White
Write-Host "  • 螢幕保護：已禁用" -ForegroundColor White
Write-Host ""

Write-Host "⚠️  重要提醒：" -ForegroundColor Yellow
Write-Host "  1. 請確保電腦接電源（筆記型電腦）" -ForegroundColor White
Write-Host "  2. 網絡連接保持穩定" -ForegroundColor White
Write-Host "  3. 不要手動休眠或關機" -ForegroundColor White
Write-Host ""

Write-Host "📝 下一步：" -ForegroundColor Cyan
Write-Host "  返回 WSL 並運行爬蟲：" -ForegroundColor White
Write-Host "  bash scripts/start_crawler_tmux.sh --mode balanced --max-videos 10" -ForegroundColor Green
Write-Host ""

Write-Host "📚 完整文檔：" -ForegroundColor Cyan
Write-Host "  查看 QUICK_START_LONG_RUNNING.md" -ForegroundColor White
Write-Host ""

Read-Host "按 Enter 鍵退出"
