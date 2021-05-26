$env:PYTHONPATH += ";$(Join-Path $(Get-Location) "..\..\my_package")"

Get-ChildItem -Filter "build" -Directory | Remove-Item -Recurse
Get-ChildItem -Filter "dist" -Directory | Remove-Item -Recurse
Get-ChildItem -Filter "*.egg-info" -Directory | Remove-Item -Recurse

python setup.py install