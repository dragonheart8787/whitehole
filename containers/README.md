# WhiteSearch Docker

## Build

From the **repository root**:

```bash
docker build -f containers/Dockerfile -t whitesearch:latest .
```

## Run

```bash
# Help
docker run --rm whitesearch:latest --help

# Mount repo as /workspace (artifacts persist on host)
docker run --rm -v "$(pwd):/workspace" -w /workspace whitesearch:latest \
  compare --model bounce --data mock --channel gw

# Compose
docker compose -f containers/docker-compose.yml run --rm ws calibrate --profile quick
```

## Verify

```powershell
.\scripts\verify-docker.ps1
```

Requires Docker Desktop with the **Linux** engine running.

## Image contents

- Python 3.11 (Debian Bookworm)
- `pip install -e ".[science,gw]"` (bilby, dynesty, gwpy, astropy, …)
- Best-effort: PyCBC, einsteinpy, ehtim
