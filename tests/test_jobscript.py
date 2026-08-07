import pytest

from gpu_queue.jobscript import JobScriptError, parse_directives, parse_size


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


# -- parse_size ------------------------------------------------------------


def test_size_gib_units():
    for text in ("12G", "12g", "12GiB", "12gib", "12GB", "12gb"):
        assert parse_size(text) == 12288, text


def test_size_mib_units():
    for text in ("512M", "512m", "512MiB", "512MB"):
        assert parse_size(text) == 512, text


def test_size_bare_number_is_mib():
    assert parse_size("8192") == 8192

# Not Implemented! #TODO:
# def test_size_fractional():
#     assert parse_size("1.5G") == 1536
#     assert parse_size("0.5G") == 512


def test_size_surrounding_whitespace_ok():
    assert parse_size("  12G  ") == 12288


def test_size_rejects_zero():
    # A zero budget would let a job onto every GPU forever.
    with pytest.raises(JobScriptError, match="> 0"):
        parse_size("0")


def test_size_rejects_rounding_down_to_zero():
    # Small enough to round to 0 MiB — must not silently become a free pass.
    with pytest.raises(JobScriptError, match="> 0"):
        parse_size("0.0001G")


def test_size_rejects_negative():
    with pytest.raises(JobScriptError, match="invalid"):
        parse_size("-4G")


def test_size_rejects_unknown_unit():
    with pytest.raises(JobScriptError, match="invalid"):
        parse_size("12TB")


def test_size_rejects_trailing_junk():
    # re.match would stop at the space and silently drop the rest.
    with pytest.raises(JobScriptError, match="invalid"):
        parse_size("12G junk")


def test_size_rejects_nonsense():
    for text in ("abc", "", "G", "1.5", "1.2.3G"):
        with pytest.raises(JobScriptError, match="invalid"):
            parse_size(text)
