' Horse Race Predictor - double-click this to open the app.
'
' Runs the launcher through pythonw.exe (no console window). The first run
' hands off to run.bat visibly, because the setup takes a few minutes.

Option Explicit

Dim fso, sh, here, pyw, launcher, setup
Set fso = CreateObject("Scripting.FileSystemObject")
Set sh  = CreateObject("WScript.Shell")

here     = fso.GetParentFolderName(WScript.ScriptFullName)
pyw      = here & "\.venv\Scripts\pythonw.exe"
launcher = here & "\launcher.py"
setup    = here & "\run.bat"

If Not fso.FileExists(launcher) Then
    MsgBox "launcher.py is missing from" & vbCrLf & here, 16, "Horse Race Predictor"
    WScript.Quit 1
End If

If Not fso.FileExists(pyw) Then
    If Not fso.FileExists(setup) Then
        MsgBox "run.bat is missing, so the first-time setup cannot run.", 16, "Horse Race Predictor"
        WScript.Quit 1
    End If
    sh.Run """" & setup & """ setup", 1, True
End If

If Not fso.FileExists(pyw) Then
    MsgBox "Setup did not finish. Run run.bat in this folder to see what went wrong.", _
           16, "Horse Race Predictor"
    WScript.Quit 1
End If

sh.Run """" & pyw & """ """ & launcher & """", 0, False
