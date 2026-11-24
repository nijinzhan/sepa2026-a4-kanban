"""Start an isolated local DB and apply pinned, unmodified A-5 migrations.

Passwords and upstream SQL stay in ignored outputs/postgres. Never resets a volume.
"""

import argparse
import hashlib
import json
import secrets
import shutil
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "outputs/postgres"
REF = "745db5cd4d9a52d8bf68adf38290e52f09904aca"
CONTAINER = "a4-lihong-postgres-20261030"
VOLUME = "a4-lihong-pg16-20261030"
OWNER = "2024141460331"
IMAGE = (
    "public.ecr.aws/docker/library/postgres@"
    "sha256:cbf42f0edc946c0bc1b8eeed62403734e1f25dc67a3491455d07a92fdb558b28"
)


def _docker_2024141460331(*args, stdin=None, check=True):
    executable = (
        shutil.which("docker") or r"C:\Program Files\Docker\Docker\resources\bin\docker.exe"
    )
    result = subprocess.run([executable, *args], input=stdin, capture_output=True, timeout=180)
    if check and result.returncode:
        # SQL/CLI errors may contain credentials: keep diagnostics local.
        STATE.mkdir(parents=True, exist_ok=True)
        (STATE / "last-error.log").write_bytes(result.stderr)
        raise RuntimeError("Docker/SQL failed; see ignored outputs/postgres/last-error.log")
    return result


def _sql_2024141460331(data: bytes, *, single_transaction=False):
    runtime = json.loads((STATE / "runtime.local.json").read_text(encoding="utf-8"))
    args = [
        "exec",
        "-i",
        runtime["container"],
        "psql",
        "-X",
        "-A",
        "-t",
        "-v",
        "ON_ERROR_STOP=1",
        "-U",
        "alpha_dba",
        "-d",
        "alpha_atc",
    ]
    if single_transaction:
        args.append("--single-transaction")
    return _docker_2024141460331(*args, stdin=data).stdout


def main_2024141460331():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-repo", type=Path, required=True, help="P-Blue Git checkout")
    parser.add_argument(
        "--port", type=int, help="existing runtime port, or 54329 for a new runtime"
    )
    args = parser.parse_args()
    if args.port is not None and not 1 <= args.port <= 65535:
        parser.error("invalid port")
    STATE.mkdir(parents=True, exist_ok=True)
    runtime_path = STATE / "runtime.local.json"
    if runtime_path.exists():
        runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
        if args.port is not None and runtime["port"] != args.port:
            raise RuntimeError("existing runtime uses a different port")
    else:
        args.port = args.port or 54329
        admin, app, viewer = (secrets.token_hex(24) for _ in range(3))
        runtime = {
            "container": CONTAINER,
            "volume": VOLUME,
            "port": args.port,
            "sourceRef": REF,
            "adminDsn": f"postgresql://alpha_dba:{admin}@127.0.0.1:{args.port}/alpha_atc",
            "appDsn": f"postgresql://alpha_app:{app}@127.0.0.1:{args.port}/alpha_atc",
            "viewerDsn": f"postgresql://alpha_viewer:{viewer}@127.0.0.1:{args.port}/alpha_atc",
            "adminPassword": admin,
            "appPassword": app,
            "viewerPassword": viewer,
        }
        runtime_path.write_text(json.dumps(runtime, indent=2), encoding="utf-8")
    container = runtime["container"]
    inspect = _docker_2024141460331("inspect", container, check=False)
    if inspect.returncode == 0:
        meta = json.loads(inspect.stdout)[0]
        if meta["Config"].get("Labels", {}).get("a4.owner") != OWNER:
            raise RuntimeError("container name belongs to another runtime")
        _docker_2024141460331("start", container)
    else:
        envfile = STATE / "postgres.local.env"
        envfile.write_text(
            "POSTGRES_USER=alpha_dba\nPOSTGRES_DB=alpha_atc\nPOSTGRES_PASSWORD="
            + runtime["adminPassword"]
            + "\n",
            encoding="ascii",
        )
        _docker_2024141460331(
            "run",
            "-d",
            "--name",
            container,
            "--label",
            f"a4.owner={OWNER}",
            "-p",
            f"127.0.0.1:{runtime['port']}:5432",
            "--env-file",
            str(envfile),
            "-v",
            f"{runtime['volume']}:/var/lib/postgresql/data",
            IMAGE,
            "-c",
            "shared_preload_libraries=pg_stat_statements",
        )
    for _ in range(40):
        ready = _docker_2024141460331(
            "exec", container, "pg_isready", "-U", "alpha_dba", "-d", "alpha_atc", check=False
        )
        if ready.returncode == 0:
            break
        time.sleep(1)
    else:
        raise RuntimeError("database is not ready")
    marker = STATE / "initialized.json"
    # Setup metadata lives outside A-5's business schemas. A migration and its
    # journal row commit together, so interruption never requires replaying DDL.
    _sql_2024141460331(
        b"CREATE SCHEMA IF NOT EXISTS a4_setup_2024141460331; "
        b"CREATE TABLE IF NOT EXISTS a4_setup_2024141460331.migration_history ("
        b"path text PRIMARY KEY, sha256 text NOT NULL, source_ref text NOT NULL, "
        b"applied_at timestamptz NOT NULL DEFAULT now());"
    )
    history = json.loads(
        _sql_2024141460331(
            b"SELECT coalesce(json_agg(h),'[]') FROM a4_setup_2024141460331.migration_history h;"
        )
    )
    applied = {row["path"]: row for row in history}
    legacy = json.loads(marker.read_text(encoding="utf-8")) if marker.exists() else None
    if legacy and legacy["sourceRef"] != REF:
        raise RuntimeError("migration baseline mismatch")
    if legacy and not applied:
        _sql_2024141460331(b"SELECT count(*) FROM asr.job;")
    files = ["infra/postgres/init/00-bootstrap.sql"]
    listing = (
        subprocess.run(
            ["git", "-C", str(args.source_repo), "ls-tree", "-r", "--name-only", REF, "migrations"],
            check=True,
            capture_output=True,
        )
        .stdout.decode()
        .splitlines()
    )
    versions = [
        p
        for p in listing
        if p.startswith("migrations/V") and int(p.split("/V")[1].split("__")[0]) <= 13
    ]
    files += sorted(versions, key=lambda p: int(p.split("/V")[1].split("__")[0]))
    files += sorted(p for p in listing if p.startswith("migrations/R__"))
    manifest = {
        "sourceRepo": "https://github.com/starshipmiyabi/P-Blue",
        "sourceRef": REF,
        "files": [],
    }
    for relative in files:
        sql = subprocess.run(
            ["git", "-C", str(args.source_repo), "show", f"{REF}:{relative}"],
            check=True,
            capture_output=True,
        ).stdout
        target = STATE / "upstream" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(sql)
        entry = {"path": relative, "sha256": hashlib.sha256(sql).hexdigest()}
        # Names/digests come from pinned Git paths, not user-provided SQL values.
        journal = (
            "\nINSERT INTO a4_setup_2024141460331.migration_history(path,sha256,source_ref) "
            f"VALUES('{relative}','{entry['sha256']}','{REF}');\n"
        ).encode()
        if relative in applied:
            if (applied[relative]["sha256"], applied[relative]["source_ref"]) != (
                entry["sha256"],
                REF,
            ):
                raise RuntimeError("applied migration checksum mismatch")
            manifest["files"].append(entry)
            print("verified", relative, flush=True)
            continue
        if legacy and entry in legacy["files"]:
            # Adopt an earlier successful local run only after its raw Git hash
            # matches; a new/partially initialized volume has no success marker.
            _sql_2024141460331(journal, single_transaction=True)
            manifest["files"].append(entry)
            continue
        if relative.endswith("00-bootstrap.sql"):
            prefix = (
                f"\\set app_password '{runtime['appPassword']}'\n"
                f"\\set viewer_password '{runtime['viewerPassword']}'\n"
            ).encode()
            _sql_2024141460331(prefix + sql + journal)
        else:
            _sql_2024141460331(sql + journal, single_transaction=True)
        manifest["files"].append(entry)
        print("applied", relative, flush=True)
    marker.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"status": "ready", "container": container, "runtimeFile": str(runtime_path)}))


if __name__ == "__main__":
    main_2024141460331()
