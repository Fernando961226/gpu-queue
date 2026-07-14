import pytest

from gpu_queue.jobscript import JobScriptError, parse_directives


def test_basic_directives():
    text = """#!/bin/bash
#GQ gpus=2
#GQ name=big-sweep
python train.py
"""
    assert parse_directives(text) == {"gpus": "2", "name": "big-sweep"}


def test_directives_stop_at_body():
    text = """#!/bin/bash
#GQ gpus=2
echo hi
#GQ name=ignored
"""
    assert parse_directives(text) == {"gpus": "2"}


def test_plain_comments_ignored():
    text = "# just a comment\n#GQ gpus=1\n"
    assert parse_directives(text) == {"gpus": "1"}


def test_quoted_values():
    text = '#GQ name="my job" workdir=/tmp\n'
    assert parse_directives(text) == {"name": "my job", "workdir": "/tmp"}


def test_unknown_key_rejected():
    with pytest.raises(JobScriptError, match="unknown"):
        parse_directives("#GQ mem=64G\n")


def test_malformed_rejected():
    with pytest.raises(JobScriptError, match="malformed"):
        parse_directives("#GQ gpus\n")


def test_bad_gpus_value():
    with pytest.raises(JobScriptError, match="integer"):
        parse_directives("#GQ gpus=two\n")


def test_empty_script():
    assert parse_directives("") == {}
