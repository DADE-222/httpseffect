@echo off
setlocal EnableExtensions EnableDelayedExpansion
title "Effects GUI — Install & Run (automatique)"

REM ---- Debug: décommente pour afficher chaque commande ----
REM set "DEBUG=1"
if defined DEBUG echo on

REM ---- Config ----
cd /d "%~dp0"
set "APP_FILE=effects_gui.pyw"
set "VENV_DIR=.venv"
set "PY_WINGET_ID=Python.Python.3.12"
set "PY_URL=https://www.python.org/ftp/python/3.12.0/python-3.12.0-amd64.exe"
set "PY_EXE=%TEMP%\python-3.12.0-amd64.exe"
set "FFMPEG_ZIP=%TEMP%\ffmpeg.zip"
set "FFMPEG_URL=https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
set "FFMPEG_TARGET=%VENV_DIR%\ffmpeg"
set "PIP_LOG=%TEMP%\pip_install.log"

echo.
echo ============================================================
echo  Installateur automatique pour Effects GUI (profil : COMPLET)
echo ============================================================
echo.

REM ---- Vérifier présence du script ----
if not exist "%APP_FILE%" (
  for /f "delims=" %%F in ('dir /b /a:-d "effects_gui*.py*" 2^>NUL') do (
    if /i "%%~nF%%~xF" neq "" set "APP_FILE=%%F"
  )
)
if not exist "%APP_FILE%" (
  echo [AVERTISSEMENT] Aucun fichier "%APP_FILE%" trouve dans ce dossier.
  echo Place ton script "%APP_FILE%" ici puis relance ce .BAT.
  pause
  exit /b 1
)

REM ---- Detecter Python system ----
set "RUN_PY="

for /f "delims=" %%P in ('where py 2^>NUL') do (
  set "RUN_PY=py"
)
if not defined RUN_PY (
  for /f "delims=" %%P in ('where python 2^>NUL') do (
    set "RUN_PY=python"
  )
)

if not defined RUN_PY (
  echo [INFO] Python non trouve. Tentative d'installation automatique...
  where winget >NUL 2>&1
  if not errorlevel 1 (
    echo [INFO] winget detecte -> tentative d'installation via winget...
    winget install -e --id %PY_WINGET_ID% --silent
  )

  REM re-check after winget attempt
  for /f "delims=" %%P in ('where py 2^>NUL') do (
    set "RUN_PY=py"
  )
  if not defined RUN_PY (
    for /f "delims=" %%P in ('where python 2^>NUL') do (
      set "RUN_PY=python"
    )
  )

  if not defined RUN_PY (
    echo [INFO] Telechargement de Python depuis %PY_URL% ...
    where curl >NUL 2>&1
    if errorlevel 1 (
      where powershell >NUL 2>&1
      if errorlevel 1 (
        echo [ERREUR] Ni curl ni powershell disponibles pour telechargement. Installe Python manuellement.
        pause & exit /b 1
      ) else (
        powershell -NoProfile -Command "Try { (New-Object System.Net.WebClient).DownloadFile('%PY_URL%', '%PY_EXE%') } Catch { exit 1 }"
        if errorlevel 1 (
          echo [ERREUR] Echec du telechargement Python via PowerShell.
          pause & exit /b 1
        )
      )
    ) else (
      curl -L -o "%PY_EXE%" "%PY_URL%" || (echo [ERREUR] Echec telechargement via curl & pause & exit /b 1)
    )

    echo [INFO] Installation silencieuse de Python (utilisateur)...
    "%PY_EXE%" /quiet InstallAllUsers=0 PrependPath=1 Include_launcher=1 Include_test=0

    REM re-check after installer
    for /f "delims=" %%P in ('where py 2^>NUL') do (
      set "RUN_PY=py"
    )
    if not defined RUN_PY (
      for /f "delims=" %%P in ('where python 2^>NUL') do (
        set "RUN_PY=python"
      )
    )
    if not defined RUN_PY (
      echo [ERREUR] Python introuvable apres installation. Ferme et rouvre ta session puis relance ce script.
      pause & exit /b 1
    )
  )
)

echo [INFO] Python detecte -> %RUN_PY%

REM ---- Creer virtualenv ----
if not exist "%VENV_DIR%" (
  echo [1/4] Creation du virtualenv "%VENV_DIR%"...
  %RUN_PY% -m venv "%VENV_DIR%" || (echo [ERREUR] Echec creation venv & pause & exit /b 1)
)

REM ---- Activer et pointer python du venv directement ----
call "%VENV_DIR%\Scripts\activate.bat" || (echo [ERREUR] Echec activation venv & pause & exit /b 1)
set "PY_VENV=%VENV_DIR%\Scripts\python.exe"
if not exist "%PY_VENV%" (
  echo [ERREUR] python du venv introuvable (%PY_VENV%).
  pause & exit /b 1
)
echo [INFO] utilisation de %PY_VENV%

REM ---- Mettre a jour pip ----
echo [2/4] Mise a jour pip, setuptools et wheel...
"%PY_VENV%" -m pip install --upgrade pip setuptools wheel > "%PIP_LOG%" 2>&1
if errorlevel 1 (
  echo [WARN] Mise a jour pip a retourne une erreur. Voir %PIP_LOG%
)

REM ---- Installer dependances (COMPLET) et logger ----
echo [3/4] Installation des dependances (COMPLET)...
"%PY_VENV%" -m pip install --no-cache-dir --prefer-binary "pillow>=10.0" "numpy>=1.25" "opencv-python>=4.8" "moviepy>=2.1" "imageio-ffmpeg>=0.6.0" "scipy>=1.10" "tqdm>=4.66" "imageio>=2.31" > "%PIP_LOG%" 2>&1

if errorlevel 1 (
  echo [ERREUR] Certaines dependances ont echoue a s'installer. Contenu du log :
  echo ---------------------------------------------------------
  type "%PIP_LOG%"
  echo ---------------------------------------------------------
  echo Relance ce script en mode administrateur, ou verifie la connexion internet.
  pause & exit /b 1
) else (
  echo [OK] dependances installees.
)

REM ---- Telechargement ffmpeg statique local pour fiabilite (optionnel) ----
if not exist "%FFMPEG_TARGET%\bin\ffmpeg.exe" (
  echo [4/4] Tentative d'installation locale de ffmpeg...
  where powershell >NUL 2>&1
  if errorlevel 1 (
    echo [WARN] PowerShell non disponible -> sauter telechargement ffmpeg.
  ) else (
    powershell -NoProfile -Command "Try { Invoke-WebRequest -Uri '%FFMPEG_URL%' -OutFile '%FFMPEG_ZIP%' -UseBasicParsing -ErrorAction Stop } Catch { exit 1 }"
    if errorlevel 1 (
      echo [WARN] Telechargement ffmpeg a echoue.
    ) else (
      powershell -NoProfile -Command "Try { Expand-Archive -Path '%FFMPEG_ZIP%' -DestinationPath '%VENV_DIR%' -Force } Catch { exit 1 }"
      for /f "delims=" %%D in ('dir /b /ad "%VENV_DIR%\ffmpeg-*" 2^>NUL') do (
        if exist "%VENV_DIR%\%%D\bin\ffmpeg.exe" (
          md "%FFMPEG_TARGET%" 2>nul
          xcopy /y /e "%VENV_DIR%\%%D\bin" "%FFMPEG_TARGET%\bin" >NUL 2>&1
        )
      )
      if exist "%FFMPEG_TARGET%\bin\ffmpeg.exe" (
        echo [INFO] ffmpeg installe localement dans %FFMPEG_TARGET%\bin
      ) else (
        echo [WARN] Extraction ffmpeg OK mais executable non trouve.
      )
      del "%FFMPEG_ZIP%" 2>nul
    )
  )
)

REM ---- Lancer l'application avec le python du venv (log si erreur) ----
if exist "%APP_FILE%" (
  echo.
  echo [RUN] Lancement de "%APP_FILE%" avec l'environnement virtuel...
  "%PY_VENV%" "%APP_FILE%"
  set "ERR=%ERRORLEVEL%"
  if not "%ERR%"=="0" (
    echo.
    echo [ERROR] L'application a termine avec un code d'erreur: %ERR%
    echo Voir le log pip si installation a echoue: %PIP_LOG%
    echo Conseil: lance ce .BAT depuis une invite en administrateur si des wheel natives ont echoue.
    pause
  )
) else (
  echo [FIN] Installation terminee. Place ton script "%APP_FILE%" dans ce dossier puis relance ce .BAT.
  pause
)

endlocal
exit /b 0
