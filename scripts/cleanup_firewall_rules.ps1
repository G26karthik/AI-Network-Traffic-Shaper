# Remove Windows Firewall rules created by predict_and_shape.py
# Run in an elevated PowerShell session.

$rules = Get-NetFirewallRule | Where-Object { $_.DisplayName -like "AI-Traffic-Shaper*" }
if ($rules.Count -eq 0) {
  Write-Host "No matching rules found."
  exit 0
}

$rules | ForEach-Object {
  Write-Host "Removing:" $_.DisplayName
  Remove-NetFirewallRule -Name $_.Name -ErrorAction SilentlyContinue
}

Write-Host "Done."
