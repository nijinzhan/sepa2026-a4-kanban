import json

import pytest

from a4_speech.cli import main


def test_missing_config_exits_with_json_error_without_traceback(tmp_path, capsys):
    result = main(["--config", str(tmp_path / "missing.json"), "check-model"])
    assert result == 2
    output = json.loads(capsys.readouterr().out)
    assert output["code"] == "CONFIG_ERROR"


def test_read_does_not_require_local_models(tmp_path, capsys, monkeypatch):
    import httpx
    from support import RID, SharedServer

    server = SharedServer()
    segment = server.add_segment(0, 1000)
    sid = segment["id"]
    server.texts[sid] = {
        **segment,
        "id": "text-id",
        "segmentId": sid,
        "recordingId": RID,
        "text": "corrected",
        "textRaw": "raw",
        "revision": 2,
        "reviewStatus": "accepted",
    }
    actual_client = httpx.Client
    monkeypatch.setattr(
        "a4_speech.cli.httpx.Client",
        lambda **kw: actual_client(transport=httpx.MockTransport(server), **kw),
    )
    monkeypatch.setenv("A4_API_KEY", "secret-for-test")
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {"a3Url": "https://team.test/api/v1/a3", "a4Url": "https://team.test/api/v1/a4"}
        ),
        encoding="utf-8",
    )
    assert main(["--config", str(config), "read", "--transcript-id", "text-id"]) == 0
    output = capsys.readouterr().out
    assert json.loads(output)["transcript"]["textRaw"] == "raw"
    assert "secret-for-test" not in output
    assert json.loads(output)["historyStatus"] == "job_context_required"


@pytest.mark.parametrize("value", ["file:///a", "https://user:secret@host/api", "relative"])
def test_invalid_endpoint_is_config_error(tmp_path, capsys, monkeypatch, value):
    monkeypatch.setenv("A4_API_KEY", "test")
    path = tmp_path / "bad.json"
    path.write_text(
        json.dumps({"a3Url": value, "a4Url": "https://team.test/api/v1/a4"}), encoding="utf-8"
    )
    assert main(["--config", str(path), "read", "--transcript-id", "id"]) == 2
    assert json.loads(capsys.readouterr().out)["code"] == "CONFIG_ERROR"


def test_missing_credential_is_explicit_without_echoing_config(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("A4_API_KEY", raising=False)
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {"a3Url": "https://team.test/api/v1/a3", "a4Url": "https://team.test/api/v1/a4"}
        ),
        encoding="utf-8",
    )
    assert main(["--config", str(path), "read", "--transcript-id", "id"]) == 2
    assert json.loads(capsys.readouterr().out)["code"] == "AUTH_ENV_MISSING"
