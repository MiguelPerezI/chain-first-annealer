# Helpers sourced by reproduce_all.sh. PY and AUDIT are supplied by the caller.
audit() {
  local audit_output
  audit_output=$("$PY" "$AUDIT" "$1") || return $?
  printf '%s\n' "$audit_output" | tail -1
}
check_incomplete() {
  "$PY" -c 'import json,sys; from pathlib import Path
p=Path(sys.argv[1]); s=json.loads((p/"summary.json").read_text())
if (s.get("status") != "incomplete" or s.get("sc_strict_bound") is not None
    or s.get("all_domains_verified") is not False
    or not 0 <= s["coverage"] < s["complex"]["num_facets"]
    or (p/"cover.txt").exists() or (p/"planners.txt").exists()):
    raise SystemExit("Invalid incomplete-run record")' "$1"
}
