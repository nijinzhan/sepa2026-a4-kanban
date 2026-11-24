"""Configured limits must reject before persistence, and permit larger valid jobs."""

import json

import httpx
import pytest
from support_2024141460331 import RID, Engine, SharedServer

from a4_speech.cli_2024141460331 import ConfigError_2024141460331, load_config_2024141460331
from a4_speech.client_2024141460331 import TeamApi
from a4_speech.service_2024141460331 import (
    ProcessingError_2024141460331,
    SpeechService_2024141460331,
)


@pytest.mark.parametrize("limit", [0, -1, True, 1.5, "500", None])
def test_invalid_limit_is_rejected_before_work(limit, tmp_path):
    with pytest.raises(ValueError, match="maxSegments"):
        SpeechService_2024141460331(object(), receipt_dir=tmp_path, max_segments=limit)


@pytest.mark.parametrize("count", [1, 2, 3])
def test_configured_boundary_and_zero_side_effects(count, tmp_path):
    server = SharedServer()
    engine = Engine(spans=[(i * 10, (i + 1) * 10) for i in range(count)])
    with httpx.Client(transport=httpx.MockTransport(server)) as http:
        api = TeamApi(http, "https://team.test/api/v1/a3", "https://team.test/api/v1/a4")
        service = SpeechService_2024141460331(api, engine, receipt_dir=tmp_path, max_segments=2)
        if count <= 2:
            result = service.transcribe_2024141460331(RID, "boundary")
            assert len(result["transcripts"]) == count
            assert server.jobs[result["jobId"]]["params"]["a4_10_20"]["maxSegments"] == 2
        else:
            with pytest.raises(ProcessingError_2024141460331) as caught:
                service.transcribe_2024141460331(RID, "boundary")
            assert caught.value.code == "JOB_SEGMENT_LIMIT"
            detail = caught.value.as_dict_2024141460331()
            assert (detail["segmentCount"], detail["maxSegments"]) == (3, 2)
            assert not server.segments and not server.jobs and not server.writes
            assert engine.calls == 0


def test_raising_limit_above_500_reaches_task_creation(tmp_path):
    server = SharedServer()
    engine = Engine(spans=[(i * 5, (i + 1) * 5) for i in range(501)])
    with httpx.Client(transport=httpx.MockTransport(server)) as http:
        api = TeamApi(
            http, "https://team.test/api/v1/a3", "https://team.test/api/v1/a4", max_segments=501
        )
        service = SpeechService_2024141460331(api, engine, receipt_dir=tmp_path, max_segments=501)
        result = service.transcribe_2024141460331(RID, "larger")
        assert result["processedSegments"] == 501
        assert len(server.jobs[result["jobId"]]["params"]["a4_10_20"]["targetSegmentIds"]) == 501


@pytest.mark.parametrize("value", [True, 0, -5, 2.5, "500"])
def test_invalid_config_limit_fails_at_load(tmp_path, value):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"maxSegments": value}), encoding="utf-8")
    with pytest.raises(ConfigError_2024141460331, match="maxSegments"):
        load_config_2024141460331(path)


def test_default_limit_remains_500_and_is_enforced(tmp_path):
    server, engine = SharedServer(), Engine(spans=[(i * 5, (i + 1) * 5) for i in range(501)])
    with httpx.Client(transport=httpx.MockTransport(server)) as http:
        api = TeamApi(http, "https://team.test/api/v1/a3", "https://team.test/api/v1/a4")
        service = SpeechService_2024141460331(api, engine, receipt_dir=tmp_path)
        with pytest.raises(ProcessingError_2024141460331) as caught:
            service.transcribe_2024141460331(RID, "default")
        assert caught.value.as_dict_2024141460331()["maxSegments"] == 500
        assert not server.writes and engine.calls == 0
