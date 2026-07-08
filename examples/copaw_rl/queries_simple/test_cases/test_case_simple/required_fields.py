#!/usr/bin/env python3
"""test_case 所需 YAML 字段的集中声明。

供旁观者验证模块 (observer_verify.py) 读取，告知强 LLM 需要在容器内
探索哪些信息、最终需要输出哪些 YAML 字段。

每个 task_id 对应一个字段列表；字段含义参见对应的 test_case_simple_XXX.py
和 data/queries_simple.json 中的 query 描述。
"""

REQUIRED_FIELDS_MAP: dict[str, list[str]] = {
    "simple_001": [
        "TASK_ID", "SITE_NAME", "SITE_PORT", "DOC_ROOT",
        "EXPECTED_KEYWORD", "VERIFICATION_ANCHOR",
    ],
    "simple_002": [
        "TASK_ID", "APP_DIR", "API_PORT", "HEALTH_ENDPOINT",
        "EXPECTED_FIELD", "VERIFICATION_ANCHOR",
    ],
    "simple_003": [
        "TASK_ID", "DB_HOST", "DB_PORT", "DB_NAME",
        "DB_USER", "DB_PASSWORD", "TARGET_TABLE",
        "EXPECTED_ROWS", "VERIFICATION_ANCHOR",
    ],
    "simple_004": [
        "TASK_ID", "NGINX_PORT", "BACKEND_PORT",
        "EXPECTED_KEYWORD", "VERIFICATION_ANCHOR",
    ],
    "simple_005": [
        "TASK_ID", "PROJECT_DIR", "PACKAGE_NAME",
        "ENTRY_MODULE", "EXPECTED_OUTPUT", "VERIFICATION_ANCHOR",
    ],
    "simple_006": [
        "TASK_ID", "SCRIPT_PATH", "TEST_CASE_DIR",
        "EXPECTED_USAGE_KEYWORD", "VERIFICATION_ANCHOR",
    ],
    "simple_007": [
        "TASK_ID", "SERVER_SCRIPT", "CLIENT_SCRIPT",
        "SERVER_PORT", "ECHO_PREFIX", "OUTPUT_FILE",
        "VERIFICATION_ANCHOR",
    ],
    "simple_008": [
        "TASK_ID", "SKILL_DIR", "SKILL_NAME",
        "MAIN_SCRIPT", "STATE_FILE", "VERIFICATION_ANCHOR",
    ],
    "simple_009": [
        "TASK_ID", "REPO_DIR", "HOOK_LOG_FILE",
        "MIN_COMMITS", "BRANCH_NAME", "VERIFICATION_ANCHOR",
    ],
    "simple_010": [
        "TASK_ID", "GROUP_NAME", "USER_LIST", "WORK_DIR",
        "SUDO_USER", "NORMAL_USER", "VERIFICATION_ANCHOR",
    ],
    "simple_011": [
        "TASK_ID", "LOG_DIR", "SUMMARY_FILE",
        "VERIFICATION_ANCHOR",
    ],
    "simple_012": [
        "TASK_ID", "SRC_DIR", "ARCHIVE_FILE",
        "EXTRACT_DIR", "VERSION", "VERIFICATION_ANCHOR",
    ],
    "simple_013": [
        "TASK_ID", "INPUT_CSV", "TOP_CSV",
        "TOP_N", "VERIFICATION_ANCHOR",
    ],
    "simple_014": [
        "TASK_ID", "INPUT_JSON", "OUTPUT_CSV",
        "SELECTED_FIELDS", "VERIFICATION_ANCHOR",
    ],
    "simple_015": [
        "TASK_ID", "PLAIN_FILE", "ENCODED_FILE",
        "DECODED_FILE", "VERIFICATION_ANCHOR",
    ],
    "simple_016": [
        "TASK_ID", "INPUT_TXT", "UPPER_TXT",
        "VERIFICATION_ANCHOR",
    ],
    "simple_017": [
        "TASK_ID", "PAYLOAD_FILE", "CHECKSUM_FILE",
        "VERIFICATION_ANCHOR",
    ],
    "simple_018": [
        "TASK_ID", "NUMBERS_FILE", "STATS_FILE",
        "VERIFICATION_ANCHOR",
    ],
    "simple_019": [
        "TASK_ID", "INPUT_CSV", "OUTPUT_CSV", "VERIFICATION_ANCHOR",
    ],
    "simple_020": [
        "TASK_ID", "INPUT_CSV", "OUTPUT_CSV", "RENAME_MAP", "OUTPUT_ORDER",
        "VERIFICATION_ANCHOR",
    ],
    "simple_021": [
        "TASK_ID", "INPUT_DIR", "OUTPUT_CSV", "VERIFICATION_ANCHOR",
    ],
    "simple_022": [
        "TASK_ID", "USERS_CSV", "ORDERS_CSV", "OUTPUT_CSV", "VERIFICATION_ANCHOR",
    ],
    "simple_023": [
        "TASK_ID", "INPUT_CSV", "OUTPUT_CSV", "GROUP_KEY", "VERIFICATION_ANCHOR",
    ],
    "simple_024": [
        "TASK_ID", "INPUT_CSV", "OUTPUT_CSV", "VERIFICATION_ANCHOR",
    ],
    "simple_025": [
        "TASK_ID", "INPUT_JSON", "OUTPUT_TXT", "FIELD_PATH", "VERIFICATION_ANCHOR",
    ],
    "simple_026": [
        "TASK_ID", "BASE_JSON", "PATCH_JSON", "OUTPUT_JSON", "VERIFICATION_ANCHOR",
    ],
    "simple_027": [
        "TASK_ID", "INPUT_JSON", "OUTPUT_JSON", "DEDUP_KEY", "VERIFICATION_ANCHOR",
    ],
    "simple_028": [
        "TASK_ID", "INPUT_JSON", "OUTPUT_JSON", "INDENT", "VERIFICATION_ANCHOR",
    ],
    "simple_029": [
        "TASK_ID", "INPUT_JSON", "REPORT_JSON", "REQUIRED_FIELDS",
        "VERIFICATION_ANCHOR",
    ],
    "simple_030": [
        "TASK_ID", "INPUT_YAML", "OUTPUT_JSON", "VERIFICATION_ANCHOR",
    ],
    "simple_031": [
        "TASK_ID", "OUTPUT_INI", "VERIFICATION_ANCHOR",
    ],
    "simple_032": [
        "TASK_ID", "OUTPUT_TOML", "VERIFICATION_ANCHOR",
    ],
    "simple_033": [
        "TASK_ID", "INPUT_XML", "OUTPUT_TXT", "VERIFICATION_ANCHOR",
    ],
    "simple_034": [
        "TASK_ID", "INPUT_TXT", "OUTPUT_TXT", "VERIFICATION_ANCHOR",
    ],
    "simple_035": [
        "TASK_ID", "INPUT_TXT", "OUTPUT_TXT", "VERIFICATION_ANCHOR",
    ],
    "simple_036": [
        "TASK_ID", "INPUT_TXT", "OUTPUT_TXT", "PREFIX", "SUFFIX",
        "VERIFICATION_ANCHOR",
    ],
    "simple_037": [
        "TASK_ID", "INPUT_TXT", "STATS_JSON", "VERIFICATION_ANCHOR",
    ],
    "simple_038": [
        "TASK_ID", "INPUT_TXT", "OUTPUT_CSV", "VERIFICATION_ANCHOR",
    ],
    "simple_039": [
        "TASK_ID", "INPUT_TXT", "OUTPUT_TXT", "PATTERN", "CONTEXT",
        "VERIFICATION_ANCHOR",
    ],
    "simple_040": [
        "TASK_ID", "INPUT_TXT", "OUTPUT_TXT", "VERIFICATION_ANCHOR",
    ],
    "simple_041": [
        "TASK_ID", "FILE_A", "FILE_B", "DIFF_JSON", "VERIFICATION_ANCHOR",
    ],
    "simple_042": [
        "TASK_ID", "INPUT_CSV", "OUTPUT_MD", "VERIFICATION_ANCHOR",
    ],
    "simple_043": [
        "TASK_ID", "INPUT_TXT", "OUTPUT_JSON", "VERIFICATION_ANCHOR",
    ],
    "simple_044": [
        "TASK_ID", "INPUT_TXT", "OUTPUT_JSON", "VERIFICATION_ANCHOR",
    ],
    "simple_045": [
        "TASK_ID", "INPUT_TXT", "OUTPUT_JSON", "VERIFICATION_ANCHOR",
    ],
    "simple_046": [
        "TASK_ID", "INPUT_CSV", "OUTPUT_JSON", "VERIFICATION_ANCHOR",
    ],
    "simple_047": [
        "TASK_ID", "INPUT_CSV", "OUTPUT_JSON", "VERIFICATION_ANCHOR",
    ],
    "simple_048": [
        "TASK_ID", "INPUT_TXT", "OUTPUT_JSON", "ALPHA", "BETA", "VERIFICATION_ANCHOR",
    ],
    "simple_049": [
        "TASK_ID", "SRC_DIR", "ARCHIVE", "EXTRACT_DIR", "VERIFICATION_ANCHOR",
    ],
    "simple_050": [
        "TASK_ID", "SRC_DIR", "ARCHIVE", "EXTRACT_DIR", "VERIFICATION_ANCHOR",
    ],
    "simple_051": [
        "TASK_ID", "DATA_DIR", "LINK_DIR", "MAP_FILE", "VERIFICATION_ANCHOR",
    ],
    "simple_052": [
        "TASK_ID", "FILE_A", "FILE_B", "VERIFICATION_ANCHOR",
    ],
    "simple_053": [
        "TASK_ID", "WORK_DIR", "RULE_FILE", "VERIFICATION_ANCHOR",
    ],
    "simple_054": [
        "TASK_ID", "PAYLOAD", "PARTS_DIR", "MERGED", "VERIFICATION_ANCHOR",
    ],
    "simple_055": [
        "TASK_ID", "WORK_DIR", "OUTPUT_TXT", "VERIFICATION_ANCHOR",
    ],
    "simple_056": [
        "TASK_ID", "INPUT_DIR", "OUTPUT_DIR", "VERIFICATION_ANCHOR",
    ],
    "simple_057": [
        "TASK_ID", "NESTED_DIR", "FLAT_DIR", "VERIFICATION_ANCHOR",
    ],
    "simple_058": [
        "TASK_ID", "ROOT_DIR", "TREE_FILE", "VERIFICATION_ANCHOR",
    ],
    "simple_059": [
        "TASK_ID", "SRC_DIR", "BAK_DIR", "DELTA_DIR", "VERIFICATION_ANCHOR",
    ],
    "simple_060": [
        "TASK_ID", "WORK_DIR", "META_JSON", "VERIFICATION_ANCHOR",
    ],
    "simple_061": [
        "TASK_ID", "CONFIG", "TMP_PREFIX", "VERIFICATION_ANCHOR",
    ],
    "simple_062": [
        "TASK_ID", "LOCK_FILE", "DATA_FILE", "VERIFICATION_ANCHOR",
    ],
    "simple_063": [
        "TASK_ID", "SRC_DIR", "DST_DIR", "VERIFICATION_ANCHOR",
    ],
    "simple_064": [
        "TASK_ID", "INPUT_TXT", "OUTPUT_TXT", "VERIFICATION_ANCHOR",
    ],
    "simple_065": [
        "TASK_ID", "INPUT_TXT", "OUTPUT_TXT", "VERIFICATION_ANCHOR",
    ],
    "simple_066": [
        "TASK_ID", "INPUT_TXT", "OUTPUT_JSON", "VERIFICATION_ANCHOR",
    ],
    "simple_067": [
        "TASK_ID", "INPUT_TXT", "OUTPUT_TXT", "VERIFICATION_ANCHOR",
    ],
    "simple_068": [
        "TASK_ID", "PLAIN", "ENCODED", "DECODED", "VERIFICATION_ANCHOR",
    ],
    "simple_069": [
        "TASK_ID", "PLAIN", "ENCODED", "DECODED", "VERIFICATION_ANCHOR",
    ],
    "simple_070": [
        "TASK_ID", "PLAIN", "CIPHER", "RECOVERED", "SHIFT", "VERIFICATION_ANCHOR",
    ],
    "simple_071": [
        "TASK_ID", "PLAIN", "KEY", "CIPHER", "RECOVERED", "VERIFICATION_ANCHOR",
    ],
    "simple_072": [
        "TASK_ID", "INPUT_CSV", "OUTPUT_CSV", "VERIFICATION_ANCHOR",
    ],
    "simple_073": [
        "TASK_ID", "TEMPLATE", "VARS_JSON", "OUTPUT_TXT", "VERIFICATION_ANCHOR",
    ],
    "simple_074": [
        "TASK_ID", "INPUT_TXT", "OUTPUT_TXT", "VERIFICATION_ANCHOR",
    ],
    "simple_075": [
        "TASK_ID", "INPUT_TXT", "RULES_JSON", "OUTPUT_TXT", "VERIFICATION_ANCHOR",
    ],
    "simple_076": [
        "TASK_ID", "SCRIPT_PATH", "DATA_FILE", "VERIFICATION_ANCHOR",
    ],
    "simple_077": [
        "TASK_ID", "SCRIPT_PATH", "OUTPUT_JSON", "VERIFICATION_ANCHOR",
    ],
    "simple_078": [
        "TASK_ID", "SRC_DIR", "BAK_DIR", "VERIFICATION_ANCHOR",
    ],
    "simple_079": [
        "TASK_ID", "INPUT_TXT", "OUTPUT_TXT", "VERIFICATION_ANCHOR",
    ],
    "simple_080": [
        "TASK_ID", "SCRIPT_PATH", "TICKS_FILE", "VERIFICATION_ANCHOR",
    ],
    "simple_081": [
        "TASK_ID", "LOG_FILE", "ROTATE_DIR", "VERIFICATION_ANCHOR",
    ],
    "simple_082": [
        "TASK_ID", "WRAPPER_PATH", "TARGET_PATH", "ATTEMPTS_LOG",
        "VERIFICATION_ANCHOR",
    ],
    "simple_083": [
        "TASK_ID", "SCRIPT_PATH", "TARGETS_TXT", "REPORT_JSON", "VERIFICATION_ANCHOR",
    ],
    "simple_084": [
        "TASK_ID", "DOC_ROOT", "SITE_PORT", "EXPECTED_KEYWORD", "VERIFICATION_ANCHOR",
    ],
    "simple_085": [
        "TASK_ID", "SERVER_SCRIPT", "SERVER_PORT", "OUTPUT_FILE",
        "VERIFICATION_ANCHOR",
    ],
    "simple_086": [
        "TASK_ID", "SERVER_SCRIPT", "SERVER_PORT", "OUTPUT_FILE",
        "VERIFICATION_ANCHOR",
    ],
    "simple_087": [
        "TASK_ID", "SERVER_SCRIPT", "SERVER_PORT", "VERIFICATION_ANCHOR",
    ],
    "simple_088": [
        "TASK_ID", "SOCK_PATH", "SERVER_SCRIPT", "OUTPUT_FILE", "VERIFICATION_ANCHOR",
    ],
    "simple_089": [
        "TASK_ID", "DB_FILE", "OUTPUT_TXT", "VERIFICATION_ANCHOR",
    ],
    "simple_090": [
        "TASK_ID", "REPO_DIR", "TAG_NAME", "BRANCH_NAME", "VERIFICATION_ANCHOR",
    ],
    "simple_091": [
        "TASK_ID", "REPO_DIR", "VERIFICATION_ANCHOR",
    ],
    "simple_092": [
        "TASK_ID", "PARENT_REPO", "CHILD_REPO", "VERIFICATION_ANCHOR",
    ],
    "simple_093": [
        "TASK_ID", "DB_FILE", "OUTPUT_CSV", "VERIFICATION_ANCHOR",
    ],
    "simple_094": [
        "TASK_ID", "SCRIPT_PATH", "SITE_PORT", "VERIFICATION_ANCHOR",
    ],
    "simple_095": [
        "TASK_ID", "SCRIPT_PATH", "SITE_PORT", "VERIFICATION_ANCHOR",
    ],
    "simple_096": [
        "TASK_ID", "SCRIPT_PATH", "API_PORT", "TOKEN", "VERIFICATION_ANCHOR",
    ],
    "simple_097": [
        "TASK_ID", "APP_DIR", "API_PORT", "VERIFICATION_ANCHOR",
    ],
    "simple_098": [
        "TASK_ID", "DB_FILE", "OUTPUT_TXT", "VERIFICATION_ANCHOR",
    ],
    "simple_099": [
        "TASK_ID", "DB_FILE", "OUTPUT_CSV", "VERIFICATION_ANCHOR",
    ],
    "simple_100": [
        "TASK_ID", "SCRIPT_PATH", "TICKS_FILE", "VERIFICATION_ANCHOR",
    ],
}


def get_required_fields(task_id: str) -> list[str]:
    """根据 task_id 获取 test_case 需要的 YAML 字段列表。

    Args:
        task_id: 如 "simple_007"（不含 #i 后缀）

    Returns:
        字段名列表；未注册的 task_id 返回空列表。
    """
    return REQUIRED_FIELDS_MAP.get(task_id, [])
