from quantum_distortion.presets import list_presets, get_preset


def test_list_presets_nonempty() -> None:
    names = list_presets()

    assert isinstance(names, list)
    assert len(names) >= 1
    assert "Chordal Noise Wash" in names


def test_get_preset_has_required_keys() -> None:
    preset = get_preset("Chordal Noise Wash")

    required = {
        "key",
        "scale",
        "snap_strength",
        "smear",
        "bin_smoothing",
        "pre_quant",
        "post_quant",
        "distortion_mode",
        "distortion_params",
        "limiter_on",
        "limiter_ceiling_db",
        "dry_wet",
    }

    assert required.issubset(set(preset.keys()))

    # Distortion params should be a dict with some expected fields
    dp = preset["distortion_params"]
    assert isinstance(dp, dict)
    for k in ["fold_amount", "bias", "drive", "warmth"]:
        assert k in dp

