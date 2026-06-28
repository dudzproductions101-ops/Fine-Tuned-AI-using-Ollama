@echo off
setlocal enabledelayedexpansion

title CodeLlama AI Assistant
color 0A

echo.
echo  =====================================================
echo   CodeLlama AI Assistant  ^|  Local ^& Private
echo  =====================================================
echo.

:: ----------------------------------------------------------
:: Find Ollama — checks PATH and all known install locations
:: ----------------------------------------------------------
set OLLAMA_EXE=

:: Check PATH first
where ollama >nul 2>&1
if %errorlevel% equ 0 (
    set OLLAMA_EXE=ollama
    goto OLLAMA_FOUND
)

:: Check every known install location
for %%P in (
    "%LOCALAPPDATA%\Programs\Ollama\ollama.exe"
    "%LOCALAPPDATA%\Ollama\ollama.exe"
    "%APPDATA%\Local\Programs\Ollama\ollama.exe"
    "%PROGRAMFILES%\Ollama\ollama.exe"
    "%PROGRAMFILES(X86)%\Ollama\ollama.exe"
    "C:\Users\%USERNAME%\AppData\Local\Programs\Ollama\ollama.exe"
    "C:\Users\%USERNAME%\AppData\Local\Ollama\ollama.exe"
) do (
    if exist %%P (
        set OLLAMA_EXE=%%P
        goto OLLAMA_FOUND
    )
)

:: Still not found
echo  [!] Ollama not found. Please install it from:
echo      https://ollama.com/download
echo.
echo  After installing, close and re-open this .bat file.
pause
exit /b 1

:OLLAMA_FOUND
echo  [OK] Ollama found: %OLLAMA_EXE%

:: ----------------------------------------------------------
:: Start Ollama server if not running
:: ----------------------------------------------------------
curl -s http://localhost:11434/api/tags >nul 2>&1
if %errorlevel% neq 0 (
    echo  [..] Starting Ollama server...
    start /B "" %OLLAMA_EXE% serve >nul 2>&1
    set WAIT=0
    :WAIT_LOOP
    timeout /t 1 /nobreak >nul
    curl -s http://localhost:11434/api/tags >nul 2>&1
    if %errorlevel% equ 0 goto SERVER_READY
    set /a WAIT+=1
    if !WAIT! lss 15 goto WAIT_LOOP
    echo  [!] Server failed to start. Try running Ollama manually first.
    pause
    exit /b 1
    :SERVER_READY
    echo  [OK] Server started.
) else (
    echo  [OK] Server already running.
)

:: ----------------------------------------------------------
:: Pick model
:: ----------------------------------------------------------
set AI_MODEL=

for %%M in (mycodellama codellama:7b-instruct codellama:7b codellama deepseek-coder:6.7b deepseek-coder qwen2.5-coder:3b tinyllama) do (
    if not defined AI_MODEL (
        %OLLAMA_EXE% list 2>nul | findstr /i "%%M" >nul 2>&1
        if !errorlevel! equ 0 set AI_MODEL=%%M
    )
)

if defined AI_MODEL (
    echo  [OK] Using model: %AI_MODEL%
    goto MODEL_READY
)

echo.
echo  No model downloaded yet. Pick one:
echo.
echo    [1] codellama:7b-instruct  ^~3.8GB  Best for code
echo    [2] deepseek-coder:6.7b    ^~3.8GB  Great alternative
echo    [3] qwen2.5-coder:3b       ^~1.9GB  Smaller, good quality
echo    [4] tinyllama               ^~637MB  Tiny, fast, basic
echo    [5] Exit
echo.
set /p MODEL_CHOICE="  Your choice (1-5): "

if "!MODEL_CHOICE!"=="1" set PULL_MODEL=codellama:7b-instruct & set AI_MODEL=codellama:7b-instruct
if "!MODEL_CHOICE!"=="2" set PULL_MODEL=deepseek-coder:6.7b   & set AI_MODEL=deepseek-coder:6.7b
if "!MODEL_CHOICE!"=="3" set PULL_MODEL=qwen2.5-coder:3b      & set AI_MODEL=qwen2.5-coder:3b
if "!MODEL_CHOICE!"=="4" set PULL_MODEL=tinyllama              & set AI_MODEL=tinyllama
if "!MODEL_CHOICE!"=="5" exit /b 0

if not defined PULL_MODEL (
    echo Invalid choice.
    pause
    exit /b 1
)

echo.
echo  [..] Downloading !PULL_MODEL! — downloads once, cached forever.
echo.
%OLLAMA_EXE% pull !PULL_MODEL!
if %errorlevel% neq 0 (
    echo  [!] Download failed. Check your internet and try again.
    pause
    exit /b 1
)

:MODEL_READY

:: ----------------------------------------------------------
:: Launch chat
:: ----------------------------------------------------------
echo.
echo  =====================================================
echo   Model  : %AI_MODEL%
echo   Private: Nothing leaves your machine
echo  =====================================================
echo.

set USE_PYTHON=0
where python >nul 2>&1
if %errorlevel% equ 0 set USE_PYTHON=1

if "!USE_PYTHON!"=="1" (
    python -c "
import sys, json, urllib.request, urllib.error, os

MODEL = r'%AI_MODEL%'
OLLAMA = r'%OLLAMA_EXE%'
API_URL = 'http://localhost:11434/api/generate'

try:
    import ctypes
    ctypes.windll.kernel32.SetConsoleMode(ctypes.windll.kernel32.GetStdHandle(-11), 7)
    G='\033[92m'; C='\033[96m'; Y='\033[93m'; R='\033[0m'; B='\033[1m'; D='\033[2m'
except:
    G=C=Y=R=B=D=''

SYSTEM = 'You are an expert coding assistant. Write clean, well-commented code. Use code blocks. Be concise but complete.'

def ask(prompt):
    body = json.dumps({'model':MODEL,'prompt':prompt,'system':SYSTEM,'stream':True,
        'options':{'temperature':0.2,'top_p':0.95,'repeat_penalty':1.1,'num_predict':1024}}).encode()
    req = urllib.request.Request(API_URL, data=body, headers={'Content-Type':'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            full = ''
            print(f'{G}AI:{R} ', end='', flush=True)
            while True:
                line = r.readline()
                if not line: break
                try:
                    chunk = json.loads(line)
                    token = chunk.get('response','')
                    print(token, end='', flush=True)
                    full += token
                    if chunk.get('done'): break
                except: pass
            print()
            return full
    except Exception as e:
        print(f'{Y}[Error] {e}{R}')
        return ''

print(f'{B}{C}  Chat ready  |  Model: {MODEL}{R}')
print(f'{D}  Type your coding question. Commands: exit, clear, help{R}')
print()
while True:
    try:
        user = input(f'{Y}You:{R} ').strip()
    except (EOFError, KeyboardInterrupt):
        print(f'\n{D}Goodbye!{R}'); break
    if not user: continue
    if user.lower() in ('exit','quit','q'): print(f'{D}Goodbye!{R}'); break
    if user.lower() == 'clear':
        os.system('cls')
        print(f'{D}[Cleared]{R}\n')
        continue
    if user.lower() == 'help':
        print(f'''{C}
  Examples:
    Write a Python function to read a CSV and return a list of dicts
    Explain what this code does: [paste code]
    Fix the bug in this JavaScript: [paste code]
    Create a REST API in FastAPI with CRUD for a todo list
    Write a SQL query to find duplicate rows in a table
{R}''')
        continue
    print()
    ask(user)
    print()
"
) else (
    echo  Python not found - using Ollama built-in chat. Type /bye to exit.
    echo.
    %OLLAMA_EXE% run %AI_MODEL% --nowordwrap
)

echo.
echo  Session ended. Run this .bat anytime to chat again.
echo.
pause
