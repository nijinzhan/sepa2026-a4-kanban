import json

import pytest

from a4_speech.cli_2024141460331 import main_2024141460331


def test_missing_config_exits_with_json_error_without_traceback(tmp_path, capsys):
    result = main_2024141460331(["--config", str(tmp_path / "missing.json"), "check-model"])
    assert result == 2
    output = json.loads(capsys.readouterr().out)
    assert output["code"] == "CONFIG_ERROR"


def test_read_does_not_require_local_models(tmp_path, capsys, monkeypatch):
    import httpx
    from support_2024141460331 import RID, SharedServer

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
        "a4_speech.cli_2024141460331.httpx.Client",
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
    assert main_2024141460331(["--config", str(config), "read", "--transcript-id", "text-id"]) == 0
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
    assert main_2024141460331(["--config", str(path), "read", "--transcript-id", "id"]) == 2
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
    assert main_2024141460331(["--config", str(path), "read", "--transcript-id", "id"]) == 2
    assert json.loads(capsys.readouterr().out)["code"] == "AUTH_ENV_MISSING"


@pytest.fixture
def cli_config(tmp_path, monkeypatch):
    monkeypatch.setenv("A4_API_KEY", "secret-for-test")
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "a3Url": "https://team.test/api/v1/a3",
                "a4Url": "https://team.test/api/v1/a4",
                "receiptDirectory": str(tmp_path / "receipts"),
            }
        ),
        encoding="utf-8",
    )
    return path


def test_transcribe_no_save_returns_result_for_explicit_later_save(cli_config, capsys, monkeypatch):
    engine = object()
    monkeypatch.setattr("a4_speech.cli_2024141460331._engine_2024141460331", lambda config: engine)

    class Service:
        def __init__(self, api, configured_engine, **kwargs):
            assert configured_engine is engine

        def transcribe_2024141460331(self, recording_id, key, business_task_id=None):
            assert (recording_id, key, business_task_id) == ("recording", "once", "task-1")
            return {"status": "partial", "saveConfirmed": False, "transcripts": []}

        def run_2024141460331(self, *args, **kwargs):
            pytest.fail("--no-save must not use the combined execution and save entry")

    monkeypatch.setattr("a4_speech.cli_2024141460331.SpeechService_2024141460331", Service)
    assert (
        main_2024141460331(
            [
                "--config",
                str(cli_config),
                "transcribe",
                "--recording-id",
                "recording",
                "--idempotency-key",
                "once",
                "--business-task-id",
                "task-1",
                "--no-save",
            ]
        )
        == 0
    )
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "partial"
    assert output["saveConfirmed"] is False


def test_transcribe_default_retains_combined_entry_with_business_task(
    cli_config, capsys, monkeypatch
):
    monkeypatch.setattr(
        "a4_speech.cli_2024141460331._engine_2024141460331", lambda config: object()
    )

    class Service:
        def __init__(self, *args, **kwargs):
            pass

        def run_2024141460331(self, recording_id, key, business_task_id=None):
            assert (recording_id, key, business_task_id) == ("recording", "once", "task-1")
            return {"saveConfirmed": True}

    monkeypatch.setattr("a4_speech.cli_2024141460331.SpeechService_2024141460331", Service)
    assert (
        main_2024141460331(
            [
                "--config",
                str(cli_config),
                "transcribe",
                "--recording-id",
                "recording",
                "--idempotency-key",
                "once",
                "--business-task-id",
                "task-1",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["saveConfirmed"] is True


def test_save_reads_result_object_without_loading_model(cli_config, tmp_path, capsys, monkeypatch):
    result = {"status": "partial", "businessTaskId": "task-1", "transcripts": []}
    result_file = tmp_path / "result.json"
    result_file.write_text(json.dumps(result), encoding="utf-8-sig")

    def no_model(config):
        pytest.fail("saving an existing result must not require local model files")

    class Service:
        def __init__(self, api, engine, **kwargs):
            assert engine is None

        def save_result_2024141460331(self, loaded_result):
            assert loaded_result == result
            return {"status": "partial", "saveConfirmed": True}

    monkeypatch.setattr("a4_speech.cli_2024141460331._engine_2024141460331", no_model)
    monkeypatch.setattr("a4_speech.cli_2024141460331.SpeechService_2024141460331", Service)
    assert (
        main_2024141460331(["--config", str(cli_config), "save", "--result-file", str(result_file)])
        == 0
    )
    output = capsys.readouterr().out
    assert json.loads(output)["saveConfirmed"] is True
    assert "secret-for-test" not in output


@pytest.mark.parametrize("contents", ["not-json", "[]", "null", '"text"'])
def test_save_rejects_non_object_result_file_without_exposing_content(
    cli_config, tmp_path, capsys, monkeypatch, contents
):
    result_file = tmp_path / "private-result.json"
    result_file.write_text(contents, encoding="utf-8")

    class Service:
        def __init__(self, *args, **kwargs):
            pass

        def save_result_2024141460331(self, result):
            pytest.fail("invalid input must not be submitted for saving")

    monkeypatch.setattr("a4_speech.cli_2024141460331.SpeechService_2024141460331", Service)
    assert (
        main_2024141460331(["--config", str(cli_config), "save", "--result-file", str(result_file)])
        == 2
    )
    output = capsys.readouterr().out
    assert json.loads(output)["code"] == "INVALID_RESULT_FILE"
    assert str(result_file) not in output
    assert "secret-for-test" not in output


def test_save_reports_missing_result_file(cli_config, tmp_path, capsys):
    assert (
        main_2024141460331(
            ["--config", str(cli_config), "save", "--result-file", str(tmp_path / "missing.json")]
        )
        == 2
    )
    assert json.loads(capsys.readouterr().out)["code"] == "INVALID_RESULT_FILE"


def test_save_can_bind_a_previously_unbound_business_task(
    cli_config, tmp_path, capsys, monkeypatch
):
    result_file = tmp_path / "result.json"
    result_file.write_text(json.dumps({"status": "succeeded", "businessTaskId": None}))

    class Service:
        def __init__(self, *args, **kwargs):
            pass

        def save_result_2024141460331(self, result):
            assert result["businessTaskId"] == "task-1"
            return {"saveConfirmed": True, "businessTaskId": result["businessTaskId"]}

    monkeypatch.setattr("a4_speech.cli_2024141460331.SpeechService_2024141460331", Service)
    assert (
        main_2024141460331(
            [
                "--config",
                str(cli_config),
                "save",
                "--result-file",
                str(result_file),
                "--business-task-id",
                "task-1",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["businessTaskId"] == "task-1"
    assert json.loads(result_file.read_text())["businessTaskId"] is None


def test_save_does_not_reassign_a_bound_business_task(cli_config, tmp_path, capsys, monkeypatch):
    result_file = tmp_path / "result.json"
    result_file.write_text(json.dumps({"status": "succeeded", "businessTaskId": "task-1"}))

    class Service:
        def __init__(self, *args, **kwargs):
            pass

        def save_result_2024141460331(self, result):
            pytest.fail("a different task must not replace the result's original association")

    monkeypatch.setattr("a4_speech.cli_2024141460331.SpeechService_2024141460331", Service)
    assert (
        main_2024141460331(
            [
                "--config",
                str(cli_config),
                "save",
                "--result-file",
                str(result_file),
                "--business-task-id",
                "task-2",
            ]
        )
        == 2
    )
    assert json.loads(capsys.readouterr().out)["code"] == "RESULT_CONTEXT_MISMATCH"


def test_cli_can_explicitly_save_partial_transcription_without_reloading_model(
    cli_config, tmp_path, capsys, monkeypatch
):
    import httpx
    from support_2024141460331 import RID, Engine, SharedServer

    server = SharedServer()
    engine = Engine(fail_at=2)
    actual_client = httpx.Client
    monkeypatch.setattr(
        "a4_speech.cli_2024141460331.httpx.Client",
        lambda **kw: actual_client(transport=httpx.MockTransport(server), **kw),
    )
    monkeypatch.setattr("a4_speech.cli_2024141460331._engine_2024141460331", lambda config: engine)
    assert (
        main_2024141460331(
            [
                "--config",
                str(cli_config),
                "transcribe",
                "--recording-id",
                RID,
                "--idempotency-key",
                "one-partial-request",
                "--business-task-id",
                "task-1",
                "--no-save",
            ]
        )
        == 0
    )
    unsaved = json.loads(capsys.readouterr().out)
    assert unsaved["status"] == "partial"
    assert unsaved["saveConfirmed"] is False
    assert unsaved["processedSegments"] == 2
    assert unsaved["failedSegments"] == 1
    assert len(unsaved["transcripts"]) == 1
    assert server.texts == {}
    result_file = tmp_path / "partial.json"
    result_file.write_text(json.dumps(unsaved), encoding="utf-8")

    def no_model(config):
        pytest.fail("the separate save must not repeat inference or reload model files")

    monkeypatch.setattr("a4_speech.cli_2024141460331._engine_2024141460331", no_model)
    assert (
        main_2024141460331(["--config", str(cli_config), "save", "--result-file", str(result_file)])
        == 0
    )
    saved = json.loads(capsys.readouterr().out)
    assert saved["status"] == "partial"
    assert saved["saveConfirmed"] is True
    assert len(server.texts) == 1
    assert next(iter(server.texts.values()))["textRaw"] == "flight 1 cleared"
    assert engine.calls == 2
