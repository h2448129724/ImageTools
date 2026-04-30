"""Tests for core.function_registry module."""
import pytest
from core.function_registry import (
    FunctionDef, FUNCTION_DEFS,
    get_function_def, get_categories, get_functions_by_category,
    get_param_specs, get_all_functions_flat, get_function_registry_dict,
)


class TestFunctionDef:
    def test_defaults(self):
        d = FunctionDef(category="cat", key="k", name="n")
        assert d.params == []
        assert d.handler_type == "simple"

    def test_custom(self):
        params = [{"name": "x", "type": "int", "default": 1}]
        d = FunctionDef("cat", "k", "n", params=params, handler_type="batch")
        assert d.params == params
        assert d.handler_type == "batch"


class TestLookupHelpers:
    def test_get_function_def_found(self):
        d = get_function_def("resize")
        assert d is not None
        assert d.key == "resize"
        assert d.category == "基础处理"

    def test_get_function_def_not_found(self):
        assert get_function_def("nonexistent_key") is None

    def test_get_categories_unique_ordered(self):
        cats = get_categories()
        assert len(cats) == len(set(cats))
        assert "基础处理" in cats
        assert "颜色转换" in cats

    def test_get_functions_by_category(self):
        funcs = get_functions_by_category("颜色转换")
        assert len(funcs) >= 14
        keys = [k for k, _ in funcs]
        assert "color_bgr2rgb" in keys

    def test_get_functions_by_category_empty(self):
        assert get_functions_by_category("不存在的分类") == []

    def test_get_param_specs_found(self):
        specs = get_param_specs("resize")
        assert len(specs) > 0
        names = [s["name"] for s in specs]
        assert "width" in names

    def test_get_param_specs_not_found(self):
        assert get_param_specs("nonexistent_key") == []

    def test_get_all_functions_flat(self):
        flat = get_all_functions_flat()
        assert len(flat) == len(FUNCTION_DEFS)
        for key, name, cat in flat:
            assert isinstance(key, str)
            assert isinstance(name, str)
            assert isinstance(cat, str)

    def test_get_function_registry_dict(self):
        reg = get_function_registry_dict()
        assert "基础处理" in reg
        assert isinstance(reg["基础处理"], list)
        assert len(reg["基础处理"]) > 0


class TestRegistryConsistency:
    def test_all_keys_unique(self):
        keys = [d.key for d in FUNCTION_DEFS]
        assert len(keys) == len(set(keys))

    def test_all_handler_types_valid(self):
        for d in FUNCTION_DEFS:
            assert d.handler_type in ("simple", "complex", "batch"), f"{d.key}: {d.handler_type}"

    def test_choice_params_have_choices(self):
        for d in FUNCTION_DEFS:
            for p in d.params:
                if p["type"] == "choice":
                    assert "choices" in p, f"{d.key}.{p['name']}: choice missing 'choices'"

    def test_combo_params_have_options(self):
        for d in FUNCTION_DEFS:
            for p in d.params:
                if p["type"] == "combo":
                    assert "options" in p, f"{d.key}.{p['name']}: combo missing 'options'"
