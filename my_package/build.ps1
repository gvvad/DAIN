$homeDir = Get-Location
$env:PYTHONPATH += ";$homeDir"

foreach ($item in $(Get-ChildItem -Directory)) {
    Set-Location $item

    Get-ChildItem -Filter "build" -Directory | Remove-Item -Recurse
    Get-ChildItem -Filter "dist" -Directory | Remove-Item -Recurse
    Get-ChildItem -Filter "*.egg-info" -Directory | Remove-Item -Recurse

    python setup.py install
    
    Set-Location $homeDir
}