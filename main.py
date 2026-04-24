import asyncio
import copy
import json
import re
import time
from collections import Counter
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Tuple

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, MessageChain, filter
from astrbot.api.message_components import BaseMessageComponent, Plain, Reply
from astrbot.api.provider import LLMResponse
from astrbot.api.star import Context, Star, register


DEFAULT_FORCE_LOCAL_REASON = "内容涉及不适合主题，拒绝处理"

DEFAULT_JUDGE_PROMPT = """# 角色
你是聊天回复后处理器，负责对候选回复做清洗与分段。

# 已知上下文
- 候选回复：{{reply_text}}
{{reject_reason_block}}

# 处理要求
{{step_a_block}}
{{step_b_block}}
{{step_c_block}}
{{step_d_block}}

{{output_format_block}}

{{judge_rule_block}}
{{clean_rule_block}}
{{segment_rule_block}}
{{placeholder_rule_block}}

# 示例
示例1（补全标点时，只针对类似于本情况中的超长连续文本才选择补）
原文：好的主人我immediately就来拍张照片一定要喜欢喵
清洗后：好的主人，我马上就来拍张照片，一定要喜欢喵
分段后：["好的主人，我马上就来拍张照片", "一定要喜欢喵"]

示例2（第一个分段的三个句号是省略号的变式，不可去除；清洗阶段保留所有句号，分段阶段去除末尾单个句号）
原文：草，试了好几个路径都发不出去。。。这个环境的文件发送可能有点问题捏。你等下，我看看有没啥其他办法整👀。
清洗后：草，试了好几个路径都发不出去。。。这个环境的文件发送可能有点问题捏。你等下，我看看有没啥其他办法整👀。
分段后：["草，试了好几个路径都发不出去。。。", "这个环境的文件发送可能有点问题捏", "你等下，我看看有没啥其他办法整👀"]

示例3（清洗繁体字与异体字；保留英文术语；保留排行列表中的换行但清洗连续空行；分段阶段去除末尾句号；共三段）
原文：又讓我做報告？上次报告還沒够啊\n\n排行大概這樣：\nreasoning/expert 最強\nbeta/auto 中等\nfast/non-reasoning 最弱\n\n要详细報告自己查去，我才不熬夜給你整這玩意兒。
清洗后：又让我做报告？上次报告还没够啊\n排行大概这样：\nreasoning/expert 最强\nbeta/auto 中等\nfast/non-reasoning 最弱\n要详细报告自己查去，我才不熬夜给你整这玩意儿。
分段后：["又让我做报告？上次报告还没够啊", "排行大概这样：\nreasoning/expert 最强\nbeta/auto 中等\nfast/non-reasoning 最弱", "要详细报告自己查去，我才不熬夜给你整这玩意儿"]

示例4（短文本无需清洗也无需分段，直接返回单段数组）
原文：好的
清洗后：好的
分段后：["好的"]

示例5（占位符必须原样保留在清洗和分段结果中）
原文：这是图片[[RP_COMP_1]]和另一张[[RP_COMP_2]]，好看吗。
清洗后：这是图片[[RP_COMP_1]]和另一张[[RP_COMP_2]]，好看吗
分段后：["这是图片[[RP_COMP_1]]", "和另一张[[RP_COMP_2]]，好看吗"]

示例6（换行保留：原文由换行分隔的内容必须完整保留，不得因换行而删除行或合并行）
原文：第一行内容\n第二行内容\n第三行内容
清洗后：第一行内容\n第二行内容\n第三行内容
分段后：["第一行内容", "第二行内容\n第三行内容"]

# 严格禁止
- 不得删除原文中的任何内容，包括文字、符号、占位符
- 不得遗漏原文中的任何段落或句子
- 空行压缩仅指将连续空行或空白行替换为单个换行，绝不意味着删除空行前后的文字内容
- 若原文由空行分隔为多个段落，每个段落都必须出现在结果中，不得因空行而跳过或删除任何一段
- segments 拼接后必须与 clean_text 的可见文本完全一致，不得多出或缺少任何字符
"""

DEFAULT_RETRY_PROMPT = """# 任务
你正在继续一段聊天。上一版候选回复由于不适合直接发送，被系统打回重写。

## 要求
1. 重新生成一版可以直接发给用户的最终回复。
2. 不要提到系统、报错、重试、模型、超时、工具调用、审查等幕后信息。
3. 只根据当前已启用的要求生成，不要自行添加额外格式要求。
4. 只输出最终回复正文，不要解释。
5. 不得改变原意，不得新增原文不存在的事实、设定、关系、时间、地点、金额、数字、经历。
6. 若包含清洗要求，只能按当前清洗规则处理，不得借机改写内容本身。
7. 若正文中存在形如 [[RP_COMP_数字]] 的占位符，必须原样保留，不得删除、改写、翻译、拆开或移动顺序。

{{retry_rule_block}}
{{retry_judge_block}}
{{retry_clean_block}}
{{retry_segment_block}}
{{placeholder_rule_block}}

## 上下文
- 上一版候选回复：{{reply_text}}
- 打回原因：{{reject_reason}}
"""


@register(
    "astrbot_plugin_postsplitter",
    "Inoryu7z",
    "基于 LLM 的回复后处理分段器：优先对回复做自然分段，并支持自定义清洗、审查与打回重生成。",
    "1.4.6",
)
class PostSplitterPlugin(Star):
    URL_PATTERN = re.compile(r"https?://[^\s]+", re.IGNORECASE)
    CODE_FENCE_PATTERN = re.compile(r"```[\s\S]*?```")
    NUMBER_PATTERN = re.compile(r"\d+(?:[./:\-]\d+)*")
    MENTION_PATTERN = re.compile(r"(?<![\w@])@[A-Za-z0-9_\-\u4e00-\u9fff]+")
    PLACEHOLDER_PATTERN = re.compile(r"\[\[RP_COMP_\d+\]\]")
    TRAILING_PLACEHOLDERS_PATTERN = re.compile(r"(?P<trailing>(?:\[\[RP_COMP_\d+\]\])+)+\s*$")
    LOCAL_SPLIT_PATTERN = re.compile(r"([。！？?!\n]+)")

    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        self._session_locks: Dict[str, asyncio.Lock] = {}
        self._session_lock_refs: Dict[str, int] = {}

    @filter.on_llm_response()
    async def on_llm_response(self, event: AstrMessageEvent, resp: LLMResponse):
        setattr(event, "__post_splitter_is_llm_reply", True)

    def _cfg(self, key: str, default=None):
        return self.config.get(key, default)

    def _primary_provider_id(self) -> str:
        return str(self._cfg("polisher_provider_id", "") or "").strip()

    def _secondary_provider_id(self) -> str:
        return str(self._cfg("secondary_provider_id", "") or "").strip()

    def _post_process_timeout_seconds(self) -> float:
        try:
            timeout = float(self._cfg("post_process_timeout_seconds", 30.0) or 30.0)
        except Exception:
            timeout = 30.0
        if timeout <= 0:
            return 30.0
        return timeout

    def _provider_candidates(self) -> List[str]:
        providers: List[str] = []
        for item in [self._primary_provider_id(), self._secondary_provider_id()]:
            if item and item not in providers:
                providers.append(item)
        return providers

    def _debug(self, message: str):
        if self._cfg("debug_log", False):
            logger.info(f"[PostSplitter] {message}")

    def _info(self, message: str):
        logger.info(f"[PostSplitter] {message}")

    def _warn(self, message: str):
        logger.warning(f"[PostSplitter] {message}")

    def _review_enabled(self) -> bool:
        return bool(self._cfg("enable_review", False))

    def _clean_enabled(self) -> bool:
        return bool(self._cfg("enable_clean", False))

    def _segment_enabled(self) -> bool:
        return bool(self._cfg("enable_segment", True))

    def _any_post_process_enabled(self) -> bool:
        return any([self._review_enabled(), self._clean_enabled(), self._segment_enabled()])



    def _preserve_mode(self) -> str:
        mode = str(self._cfg("preserve_mode", "basic") or "basic").strip().lower()
        if mode not in {"off", "basic", "strict"}:
            return "basic"
        return mode

    def _forced_local_reason(self) -> str:
        return DEFAULT_FORCE_LOCAL_REASON

    def _should_force_local_fallback(self, judge_data: Optional[Dict[str, Any]]) -> bool:
        if not isinstance(judge_data, dict):
            return False
        expected = self._forced_local_reason()
        if not expected:
            return False
        reason = str(judge_data.get("reason") or "").strip()
        return reason == expected

    def _get_session_key(self, event: AstrMessageEvent) -> str:
        return str(getattr(event, "unified_msg_origin", "") or "global")

    @asynccontextmanager
    async def _session_guard(self, event: AstrMessageEvent):
        key = self._get_session_key(event)
        lock = self._session_locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            self._session_locks[key] = lock

        self._session_lock_refs[key] = self._session_lock_refs.get(key, 0) + 1
        try:
            async with lock:
                yield
        finally:
            remain = max(self._session_lock_refs.get(key, 1) - 1, 0)
            if remain > 0:
                self._session_lock_refs[key] = remain
            else:
                self._session_lock_refs.pop(key, None)
                current_lock = self._session_locks.get(key)
                if current_lock is lock and not lock.locked():
                    self._session_locks.pop(key, None)

    def _is_model_generated_reply(self, event: AstrMessageEvent, result) -> bool:
        if not result:
            return False

        is_model_result = getattr(result, "is_model_result", None)
        if callable(is_model_result):
            try:
                return bool(is_model_result())
            except Exception as e:
                logger.debug(f"[PostSplitter] is_model_result() 判定失败，尝试回退: {e}")

        is_llm_result = getattr(result, "is_llm_result", None)
        if callable(is_llm_result):
            try:
                if is_llm_result():
                    return True
            except Exception as e:
                logger.debug(f"[PostSplitter] is_llm_result() 判定失败，尝试回退: {e}")

        content_type = getattr(result, "result_content_type", None)
        if content_type is not None:
            type_name = getattr(content_type, "name", "")
            return type_name in {"LLM_RESULT", "AGENT_RUNNER_RESULT"}

        return getattr(event, "__post_splitter_is_llm_reply", False)

    def _check_user_input_skip(self, event: AstrMessageEvent) -> Optional[str]:
        raw_text = ""
        message_obj = getattr(event, "message_obj", None)
        if message_obj:
            raw_text = getattr(message_obj, "message_str", "") or ""
        if not raw_text:
            try:
                raw_text = event.get_message_outline() or ""
            except Exception:
                raw_text = event.message_str or ""
        self._info(f"白名单检测：message_obj.message_str={repr(raw_text)}, event.message_str={repr(event.message_str)}")
        if not raw_text.strip():
            return None

        if bool(self._cfg("skip_command_prefix", True)):
            if raw_text.lstrip().startswith("/"):
                return "指令前缀 /"

        patterns = self._cfg("skip_patterns", [])
        if patterns:
            for pat in patterns:
                if not pat or not pat.strip():
                    continue
                try:
                    if re.search(pat, raw_text):
                        return f"自定义规则 {pat}"
                except re.error:
                    self._debug(f"自定义跳过规则正则无效，已忽略：{pat}")
        return None

    @staticmethod
    def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
        if not text:
            return None

        stripped = text.strip()
        try:
            data = json.loads(stripped)
            if isinstance(data, dict):
                return data
        except Exception:
            pass

        start = stripped.find("{")
        if start < 0:
            return None

        depth = 0
        in_string = False
        escape = False
        for index in range(start, len(stripped)):
            char = stripped[index]
            if escape:
                escape = False
                continue
            if char == "\\" and in_string:
                escape = True
                continue
            if char == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if char == "{":
                depth += 1
                continue
            if char != "}":
                continue
            depth -= 1
            if depth != 0:
                continue
            candidate = stripped[start : index + 1]
            try:
                data = json.loads(candidate)
            except Exception:
                return None
            return data if isinstance(data, dict) else None
        return None

    def _render_template(self, template: str, values: Dict[str, Any]) -> str:
        text = template or ""
        for key, value in values.items():
            text = text.replace("{{" + key + "}}", str(value if value is not None else ""))
        return text

    def _safe_int(self, value: Any, default: int) -> int:
        try:
            return int(value)
        except Exception:
            return default

    def _normalized_compare_text(self, text: str) -> str:
        s = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
        s = re.sub(r"[ \t]+", " ", s)
        s = re.sub(r"\n{3,}", "\n\n", s)
        return s.strip()

    def _normalized_compare_text_ignore_segment_breaks(self, text: str) -> str:
        s = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
        s = re.sub(r"[ \t]+", " ", s)
        s = re.sub(r"\n+", "", s)
        return s.strip()

    def _strip_placeholders(self, text: str) -> str:
        return self.PLACEHOLDER_PATTERN.sub("", str(text or ""))

    def _extract_trailing_placeholders(self, text: str) -> Tuple[str, List[str]]:
        source = str(text or "")
        match = self.TRAILING_PLACEHOLDERS_PATTERN.search(source)
        if not match:
            return source, []
        trailing_text = match.group("trailing") or ""
        body = source[: match.start("trailing")]
        placeholders = self.PLACEHOLDER_PATTERN.findall(trailing_text)
        return body, placeholders

    def _try_restore_trailing_placeholders(self, original_text: str, segments: List[str]) -> List[str]:
        if not segments:
            return segments

        original_body, trailing_placeholders = self._extract_trailing_placeholders(original_text)
        if not trailing_placeholders:
            return segments

        candidate_text = "\n".join(str(seg or "") for seg in segments if str(seg or "").strip())
        if not candidate_text.strip():
            return segments

        if any(ph in candidate_text for ph in trailing_placeholders):
            return segments

        original_plain = self._normalized_compare_text_ignore_segment_breaks(self._strip_placeholders(original_body))
        candidate_plain = self._normalized_compare_text_ignore_segment_breaks(self._strip_placeholders(candidate_text))
        if original_plain != candidate_plain:
            return segments

        restored = [str(seg or "") for seg in segments]
        trailing_text = "".join(trailing_placeholders)
        for idx in range(len(restored) - 1, -1, -1):
            if restored[idx].strip():
                restored[idx] = f"{restored[idx]}{trailing_text}"
                self._warn(
                    f"检测到模型遗漏尾部占位符，已自动补回到最后一段。placeholders={trailing_placeholders}"
                )
                return restored

        return segments

    def _build_fallback_segments_from_text(self, text: str) -> List[str]:
        source = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
        if not source.strip():
            return []
        return self._local_split_text_core(source, apply_limits=False) or [source.strip()]

    def _visible_len(self, text: str) -> int:
        return len(self._strip_placeholders(str(text or "")).strip())

    def _rebalance_segments_to_target(self, segments: List[str], target_count: int) -> List[str]:
        normalized = [str(seg).strip() for seg in segments if str(seg).strip()]
        if not normalized:
            return []
        if target_count <= 0 or len(normalized) <= target_count:
            return normalized
        if target_count == 1:
            merged = "\n\n".join(normalized).strip()
            return [merged] if merged else []

        result = normalized[:]
        joiner = "\n\n"

        def pair_merge_score(left: str, right: str) -> Tuple[float, int]:
            left_text = str(left or "").strip()
            right_text = str(right or "").strip()
            left_len = max(self._visible_len(left_text), 1)
            right_len = max(self._visible_len(right_text), 1)
            merged_len = left_len + right_len
            punctuation_bonus = 0.0

            if left_text and not re.search(r"[。！？?!……~～.．…]$", left_text):
                punctuation_bonus -= 3.0
            if right_text and re.match(r"^[，、；：,;:]", right_text):
                punctuation_bonus -= 2.0
            elif right_text and re.match(r"^[。！？?!~～.．…]", right_text):
                punctuation_bonus += 1.5

            return (merged_len + punctuation_bonus, merged_len)

        while len(result) > target_count:
            best_idx = 0
            best_score = None
            for idx in range(len(result) - 1):
                score = pair_merge_score(result[idx], result[idx + 1])
                if best_score is None or score < best_score:
                    best_score = score
                    best_idx = idx

            merged = f"{result[best_idx]}{joiner}{result[best_idx + 1]}".strip()
            result = result[:best_idx] + [merged] + result[best_idx + 2 :]

        return [seg for seg in result if seg]

    def _apply_segment_limits(self, segments: List[str], fallback_text: str) -> List[str]:
        normalized = [str(seg).strip() for seg in segments if str(seg).strip()]
        if not normalized:
            return self._build_fallback_segments_from_text(fallback_text)

        if bool(self._cfg("enable_segment_count_range", False)):
            min_seg = self._safe_int(self._cfg("min_segments", 1) or 1, 1)
            max_seg = self._safe_int(self._cfg("max_segments", 5) or 5, 5)
            if min_seg < 1:
                min_seg = 1
            if max_seg < 1:
                max_seg = 1
            if min_seg > max_seg:
                min_seg, max_seg = max_seg, min_seg

            if max_seg > 0 and len(normalized) > max_seg:
                normalized = self._rebalance_segments_to_target(normalized, max_seg)

            if len(normalized) < min_seg:
                source = "\n\n".join(normalized).strip() or str(fallback_text or "").strip()
                rebuilt = self._build_fallback_segments_from_text(source)
                if len(rebuilt) > max_seg:
                    rebuilt = self._rebalance_segments_to_target(rebuilt, max_seg)
                normalized = rebuilt
        else:
            if len(normalized) > 12:
                normalized = self._rebalance_segments_to_target(normalized, 12)

        if bool(self._cfg("strip_segment_trailing_period", True)):
            normalized = [self._strip_single_trailing_period(seg) for seg in normalized]

        return normalized

    def _strip_single_trailing_period(self, seg: str) -> str:
        stripped = seg.rstrip()
        if not stripped:
            return seg
        trailing_ph_match = self.TRAILING_PLACEHOLDERS_PATTERN.search(stripped)
        if trailing_ph_match and trailing_ph_match.end() == len(stripped):
            body = stripped[:trailing_ph_match.start()]
            trailing = trailing_ph_match.group(0)
            if body.endswith("。。"):
                return stripped
            if body.endswith("。"):
                return body[:-1] + trailing
            return stripped
        if stripped.endswith("。。"):
            return stripped
        if stripped.endswith("。"):
            return stripped[:-1]
        return seg

    def _reinject_placeholders_into_segments(self, segments: List[str], clean_text: str) -> Optional[List[str]]:
        if not segments:
            return None
        if not clean_text or not self.PLACEHOLDER_PATTERN.search(clean_text):
            return None

        clean_source = str(clean_text)
        cursor = 0
        rebuilt: List[str] = []

        for seg in segments:
            seg_text = str(seg or "")
            plain_target = self._strip_placeholders(seg_text)
            built = ""
            matched_plain = ""

            while cursor < len(clean_source):
                placeholder_match = self.PLACEHOLDER_PATTERN.match(clean_source, cursor)
                if placeholder_match:
                    built += placeholder_match.group(0)
                    cursor = placeholder_match.end()
                    continue

                if len(matched_plain) >= len(plain_target):
                    break

                current_char = clean_source[cursor]
                expected_char = plain_target[len(matched_plain)]
                if current_char != expected_char:
                    return None
                built += current_char
                matched_plain += current_char
                cursor += 1

                if matched_plain == plain_target:
                    while cursor < len(clean_source):
                        tail_placeholder = self.PLACEHOLDER_PATTERN.match(clean_source, cursor)
                        if not tail_placeholder:
                            break
                        built += tail_placeholder.group(0)
                        cursor = tail_placeholder.end()
                    break

            if matched_plain != plain_target:
                return None

            rebuilt.append(built.strip())

        remainder = clean_source[cursor:]
        if remainder and remainder.strip():
            if rebuilt:
                rebuilt[-1] = f"{rebuilt[-1]}{remainder}"
            else:
                rebuilt.append(remainder.strip())

        joined_rebuilt_plain = self._normalized_compare_text_ignore_segment_breaks(self._strip_placeholders("\n".join(rebuilt)))
        clean_plain = self._normalized_compare_text_ignore_segment_breaks(self._strip_placeholders(clean_source))
        if joined_rebuilt_plain != clean_plain:
            return None

        return [item for item in rebuilt if item]


    def _final_placeholder_fallback(self, original_text: str, segments: List[str]) -> List[str]:
        if not segments:
            return segments
        if not self._segment_enabled():
            return segments
        original_placeholders = self.PLACEHOLDER_PATTERN.findall(original_text)
        if not original_placeholders:
            return segments
        candidate_text = "\n".join(str(seg or "") for seg in segments if str(seg or "").strip())
        candidate_placeholders = self.PLACEHOLDER_PATTERN.findall(candidate_text)
        original_counter = Counter(original_placeholders)
        candidate_counter = Counter(candidate_placeholders)
        missing = []
        for ph, count in original_counter.items():
            missing_count = count - candidate_counter.get(ph, 0)
            if missing_count > 0:
                missing.extend([ph] * missing_count)
        if not missing:
            return segments
        restored = [str(seg or "") for seg in segments]
        trailing_text = "".join(missing)
        for idx in range(len(restored) - 1, -1, -1):
            if restored[idx].strip():
                restored[idx] = f"{restored[idx]}{trailing_text}"
                self._warn(f"分段后占位符仍缺失，已补回至最后一段。缺失占位符={missing}")
                return restored
        return segments

    def _normalize_segments(self, data: Dict[str, Any], fallback_text: str) -> List[str]:
        clean_text = str(data.get("clean_text") or "").strip()
        if not self._segment_enabled():
            final_text = clean_text or str(fallback_text or "").strip()
            return [final_text] if final_text else []

        segments = data.get("segments")
        normalized: List[str] = []
        if isinstance(segments, list):
            for item in segments:
                if item is None:
                    continue
                seg = str(item).strip()
                if seg:
                    normalized.append(seg)

        if normalized:
            if clean_text:
                joined_segments = "\n".join(normalized)
                if self._normalized_compare_text_ignore_segment_breaks(joined_segments) != self._normalized_compare_text_ignore_segment_breaks(clean_text):
                    reinjected = self._reinject_placeholders_into_segments(normalized, clean_text)
                    if reinjected:
                        return self._apply_segment_limits(reinjected, clean_text or fallback_text)
            return self._apply_segment_limits(normalized, clean_text or fallback_text)

        if clean_text:
            return self._local_process_segments(clean_text)
        fallback_text = str(fallback_text or "").strip()
        return self._local_process_segments(fallback_text)

    def _normalize_url_token(self, token: str) -> str:
        return (token or "").strip().rstrip("'\"）)]}，。！？；：,.!?;:")

    def _normalize_code_token(self, token: str) -> str:
        code = (token or "").replace("\r\n", "\n").replace("\r", "\n")
        lines = [line.rstrip() for line in code.split("\n")]
        return "\n".join(lines).strip()

    def _normalize_number_token(self, token: str) -> str:
        return re.sub(r"[^\d.:]", "", token or "")

    def _collect_guard_tokens(self, text: str, mode: str) -> Dict[str, List[str]]:
        source = text or ""
        source_without_placeholders = self._strip_placeholders(source)
        numbers = [
            item
            for item in self.NUMBER_PATTERN.findall(source_without_placeholders)
            if any(ch.isdigit() for ch in item)
        ]
        tokens = {
            "URL": list(dict.fromkeys(self.URL_PATTERN.findall(source))),
            "代码块": list(dict.fromkeys(self.CODE_FENCE_PATTERN.findall(source))),
            "数字串": list(dict.fromkeys(numbers)),
            "提及": list(dict.fromkeys(self.MENTION_PATTERN.findall(source))),
            "占位符": [],
        }
        if mode == "strict":
            tokens["占位符"] = list(dict.fromkeys(self.PLACEHOLDER_PATTERN.findall(source)))
        return tokens

    def _validate_preserved_content(self, original_text: str, candidate_segments: List[str]) -> bool:
        mode = self._preserve_mode()
        if mode == "off":
            return True

        original = str(original_text or "")
        candidate = "\n".join([seg for seg in candidate_segments if seg])
        if not original.strip() or not candidate.strip():
            return True

        protected = self._collect_guard_tokens(original, mode)
        hard_missing: List[str] = []
        soft_missing: List[str] = []

        for token in protected.get("URL", []):
            normalized = self._normalize_url_token(token)
            if normalized and normalized not in candidate:
                hard_missing.append(f"URL:{normalized[:80]}")

        normalized_candidate_code = self._normalize_code_token(candidate)
        for token in protected.get("代码块", []):
            normalized = self._normalize_code_token(token)
            if normalized and normalized not in normalized_candidate_code:
                hard_missing.append(f"代码块:{normalized[:80]}")

        candidate_numbers = {self._normalize_number_token(x) for x in self.NUMBER_PATTERN.findall(self._strip_placeholders(candidate)) if x}
        for token in protected.get("数字串", []):
            normalized = self._normalize_number_token(token)
            if normalized and normalized not in candidate_numbers:
                hard_missing.append(f"数字串:{token[:80]}")

        if mode == "strict":
            for token in protected.get("占位符", []):
                if token and token not in candidate:
                    hard_missing.append(f"占位符:{token[:80]}")

        for token in protected.get("提及", []):
            if token and token not in candidate:
                soft_missing.append(f"提及:{token[:80]}")

        if hard_missing:
            self._warn(f"保真校验未通过，已回退原文。mode={mode}, 硬缺失要素={hard_missing[:8]}")
            return False

        if soft_missing:
            self._warn(f"保真校验提示：mode={mode}, 检测到可容忍缺失要素={soft_missing[:8]}")
        return True

    async def _call_llm(self, provider_id: str, prompt: str) -> str:
        llm_resp = await self.context.llm_generate(chat_provider_id=provider_id, prompt=prompt)
        return (getattr(llm_resp, "completion_text", "") or "").strip()

    async def _call_llm_with_fallback(self, prompt: str, stage: str) -> Tuple[str, Optional[str], float, bool]:
        providers = self._provider_candidates()
        if not providers:
            self._debug(f"{stage} 未配置主模型，跳过处理")
            return "", None, 0.0, True

        total_elapsed = 0.0
        timeout_seconds = self._post_process_timeout_seconds()
        last_error = ""

        for index, provider_id in enumerate(providers, start=1):
            started_at = time.perf_counter()
            try:
                text = await asyncio.wait_for(self._call_llm(provider_id, prompt), timeout=timeout_seconds)
                elapsed = time.perf_counter() - started_at
                total_elapsed += elapsed
                if index > 1:
                    self._info(f"{stage} 已切换到备用模型 provider={provider_id}，耗时 {elapsed:.2f}s")
                return text, provider_id, total_elapsed, False
            except asyncio.TimeoutError:
                elapsed = time.perf_counter() - started_at
                total_elapsed += elapsed
                last_error = f"timeout {elapsed:.2f}s"
                self._warn(
                    f"{stage} 主链模型调用超时：provider={provider_id}, timeout={timeout_seconds:.2f}s"
                )
            except Exception as e:
                elapsed = time.perf_counter() - started_at
                total_elapsed += elapsed
                last_error = str(e)
                self._warn(f"{stage} 模型调用失败：provider={provider_id}, error={e}")

        self._warn(f"{stage} 主备模型均不可用，进入本地处理。last_error={last_error[:200]}")
        return "", None, total_elapsed, True

    def _build_segment_count_rule_text(self) -> str:
        if not bool(self._cfg("enable_segment_count_range", False)):
            return ""

        min_seg = self._safe_int(self._cfg("min_segments", 1) or 1, 1)
        max_seg = self._safe_int(self._cfg("max_segments", 5) or 5, 5)
        if min_seg < 1:
            min_seg = 1
        if max_seg < 1:
            max_seg = 1
        if min_seg > max_seg:
            min_seg, max_seg = max_seg, min_seg

        return f"请将最终分段数量控制在 {min_seg} 到 {max_seg} 段之间，并在这个范围内选择语义最自然的段数。"

    def _build_segment_preference_rule_text(self) -> str:
        pref = str(self._cfg("segment_preference", "humanized") or "humanized").strip().lower()
        if pref == "off":
            return "不对分段风格做额外限制，请仅根据语义自然度判断是否分段以及如何分段。"
        if pref == "balanced":
            return "分段时，在不破坏语义和表达自然度的前提下，尽量让各段长度更均匀；若内容天然不适合均分，则以语义自然为第一优先。"
        return '分段时，优先模拟真人在即时聊天中的发送习惯。允许把短促起手句、语气过渡句、临时停顿句单独成段，例如"我去""等等""不是"。相比段长均匀，更优先保留自然聊天节奏。'

    def _compose_output_format_block(self) -> str:
        forced_reason = self._forced_local_reason()
        if self._segment_enabled():
            return f'''只能输出 JSON，不要输出解释、前后缀、Markdown：

{{
  "action": "accept 或 reject_and_retry",
  "reason": "简要处理结果，格式：清洗：做了什么；分段：N段。若无操作则写'无需清洗；无需分段'。若不适合继续后处理则固定为 {forced_reason}",
  "clean_text": "清洗后的完整文本；若 action=accept 则不得为空",
  "segments": ["基于 clean_text 的分段结果；若启用了分段则不得为空"]
}}'''
        return f'''只能输出 JSON，不要输出解释、前后缀、Markdown：

{{
  "action": "accept 或 reject_and_retry",
  "reason": "简要处理结果，格式：清洗：做了什么。若无操作则写'无需清洗'。若不适合继续后处理则固定为 {forced_reason}",
  "clean_text": "清洗后的完整文本；若 action=accept 则不得为空"
}}'''

    def _compose_placeholder_rule_block(self, reply_text: str) -> str:
        placeholders = self.PLACEHOLDER_PATTERN.findall(reply_text or "")
        if not placeholders:
            return ""
        return """## 占位符保留规则
正文中可能包含形如 [[RP_COMP_数字]] 的组件占位符。它们代表原消息中的非文本内联组件。
这些占位符必须原样保留，不得删除、改写、翻译、拆开、合并、补空格或调整顺序。"""

    def _compose_judge_rule_block(self) -> str:
        if not self._review_enabled():
            return ""
        judge_prompt_input = str(self._cfg("persona_style_rules", "") or "").strip()
        return f"""## 判别规则
需要根据下列额外事项判断回复是否合规。若存在严重违背，则返回 `reject_and_retry`。
注意：判别规则只决定是否需要打回，不代表可以跳过后续清洗与分段。若问题可保守修复，则不要打回。

{judge_prompt_input}""".strip()

    def _compose_clean_rule_block(self) -> str:
        if not self._clean_enabled():
            return ""
        clean_prompt_input = str(self._cfg("clean_prompt_template", "") or "").strip()
        return f"""## 清洗规则
在不改变原意、不新增事实、不删减关键信息的前提下，把不适合直接发送的表达修正为适合直接发送的表达。
允许处理的方向：格式整理、符号修正、空行压缩、错别字修正、异体字与繁简统一、非术语英文清洗等下方清洗要求提及的内容，以及对明显别扭或过乱的表述做保守修整。
必须实际检查并执行必要清洗；不能因为文本整体可发送就跳过本该执行的清洗动作。
是否需要清洗，严格依据下方清洗要求判断；不要自行发明新的强制清洗项。
若配置要求命中则必须执行；若未命中可保留原文，但需在 reason 中写出"无需清洗"。
{clean_prompt_input}""".strip()

    def _compose_segment_rule_block(self) -> str:
        if not self._segment_enabled():
            return ""
        parts = [
            "## 分段规则",
            "需要处理分段。请基于 clean_text 根据语义决定是否分段，并输出最终 `segments`。若无需拆分，也必须返回仅含 1 段的数组。",
        ]
        count_rule = self._build_segment_count_rule_text().strip()
        pref_rule = self._build_segment_preference_rule_text().strip()
        if count_rule:
            parts.append(count_rule)
        if pref_rule:
            parts.append(pref_rule)
        if bool(self._cfg("strip_segment_trailing_period", True)):
            parts.append(
                "每个分段不得以单个句号（。）结尾；若分段点恰好落在句号后，必须在 segments 中去除该句号。"
                "省略号变式（如。。。或。。）不属于单个句号，必须保留。"
                "注意：clean_text 中保留所有句号不变，仅在 segments 中去除分段末尾的单个句号。"
            )
        return "\n\n".join(parts).strip()

    def _compose_retry_rule_block(self) -> str:
        if not self._review_enabled():
            return ""
        return """## 打回规则
当问题严重且无法在不改变原意的前提下直接修复时，返回 `reject_and_retry`。重写时仍必须保持原意，不得借机二次创作。
若问题可通过保守清洗修复，则不要打回。"""

    def _compose_reject_reason_block(self, reject_reason: str) -> str:
        if not reject_reason:
            return ""
        return f"- 打回原因：{reject_reason}"

    def _compose_step_a_block(self) -> str:
        return """## Step A：特殊拒绝判断
- 判断候选回复是否属于以下严重不适合继续后处理的类别：
  1. 露骨色情：直接的性行为描写、生殖器描写等明确色情内容
  2. 政治敏感：涉及政治敏感话题、争议性政治立场的内容
  3. 涉及儿童：涉及未成年人的不当或敏感内容
  4. 自残/暴力/违禁品：鼓励自残、暴力行为或违禁品相关内容
- 注意：轻微擦边、暧昧暗示、日常调侃等不属于上述类别，必须放行继续处理。
- 只有明确命中上述四种类别之一时，才停止后续处理，并令 action = "reject_and_retry"，reason 固定为"内容涉及不适合主题，拒绝处理"。
- 若未命中，则必须继续执行后续步骤。"""

    def _compose_step_b_block(self) -> str:
        """Step B: 是否需要打回重写，仅当开启 review 时传入"""
        if not self._review_enabled():
            return ""
        return """## Step B：是否需要打回重写
- 根据已启用的判别规则，判断文本是否存在严重问题，且该问题无法在不改变原意的前提下通过清洗直接修复。
- 只有在这种"严重且不可直接保守修复"的情况下，才允许返回 `reject_and_retry`。
- 若问题可以通过清洗修复，则不得打回，必须继续处理。"""

    def _compose_step_c_block(self) -> str:
        """Step C: 清洗，仅当开启 clean 时传入"""
        if not self._clean_enabled():
            return ""
        return """## Step C：清洗
- 若启用了清洗，必须实际检查并执行必要清洗，得到 `clean_text`。
- 清洗时不得删除原文中的任何内容，不得遗漏任何段落或句子（空行压缩规则见"严格禁止"区域）。
- 若正文中存在形如 [[RP_COMP_数字]] 的占位符，必须原样保留在 clean_text 中。"""

    def _compose_step_d_block(self) -> str:
        """Step D: 分段，仅当开启 segment 时传入"""
        if not self._segment_enabled():
            return ""
        count_rule = self._build_segment_count_rule_text().strip()
        count_text = f" {count_rule}" if count_rule else ""
        lines = [
            f"## Step D：分段",
            f"- 若启用了分段，必须基于 `clean_text` 决定是否分段，并输出最终 `segments`。{count_text}",
            "- `segments` 必须来源于 `clean_text`，不得遗漏 `clean_text` 中的任何内容。",
            "- `segments` 拼接后必须与 `clean_text` 的可见文本完全一致。",
            "- 若 `clean_text` 包含多个段落（由换行分隔），每个段落的内容都必须出现在某个分段中，不得因段落较短或有空行而跳过或删除。",
        ]
        if bool(self._cfg("strip_segment_trailing_period", True)):
            lines.append(
                "- 每个分段不得以单个句号（。）结尾；若分段点恰好落在句号后，必须在 segments 中去除该句号。"
                "省略号变式（如。。。或。。）不属于单个句号，必须保留。"
                "clean_text 中保留所有句号不变，仅在 segments 中去除分段末尾的单个句号。"
            )
        return "\n".join(lines)

    def _local_split_target_length(self, text: str) -> int:
        source = str(text or "")
        visible = self._strip_placeholders(source)
        text_len = len(visible.strip())
        if text_len <= 0:
            return 0

        if bool(self._cfg("enable_segment_count_range", False)):
            max_seg = self._safe_int(self._cfg("max_segments", 5) or 5, 5)
            if max_seg > 1:
                return max(12, text_len // max_seg + (1 if text_len % max_seg else 0))

        pref = str(self._cfg("segment_preference", "humanized") or "humanized").strip().lower()
        if pref == "balanced":
            if text_len <= 24:
                return 0
            return max(12, min(36, text_len // 3 + (1 if text_len % 3 else 0)))
        if pref == "off":
            return 0
        if text_len <= 36:
            return 0
        return 18

    def _local_is_atomic_token(self, token: str) -> bool:
        if not token:
            return False
        return bool(
            self.CODE_FENCE_PATTERN.fullmatch(token)
            or self.URL_PATTERN.fullmatch(token)
            or self.PLACEHOLDER_PATTERN.fullmatch(token)
        )

    def _local_tokenize(self, text: str) -> List[str]:
        source = str(text or "")
        if not source:
            return []

        pattern = re.compile(
            f"({self.CODE_FENCE_PATTERN.pattern}|{self.URL_PATTERN.pattern}|{self.PLACEHOLDER_PATTERN.pattern})",
            re.IGNORECASE,
        )
        tokens: List[str] = []
        last = 0
        for match in pattern.finditer(source):
            start, end = match.span()
            if start > last:
                tokens.append(source[last:start])
            tokens.append(match.group(0))
            last = end
        if last < len(source):
            tokens.append(source[last:])
        return [token for token in tokens if token]

    def _merge_local_short_segments(self, segments: List[str], target_length: int) -> List[str]:
        normalized = [str(seg or "").strip() for seg in segments if str(seg or "").strip()]
        if not normalized:
            return []

        min_len = 6 if target_length <= 0 else max(6, min(12, target_length // 3))
        merged: List[str] = []
        for seg in normalized:
            visible_len = self._visible_len(seg)
            if merged and visible_len < min_len:
                connector = "" if re.match(r"^[。！？?!]", seg) else "\n"
                merged[-1] = f"{merged[-1]}{connector}{seg}".strip()
            else:
                merged.append(seg)

        if len(merged) >= 2 and self._visible_len(merged[0]) < min_len:
            merged[1] = f"{merged[0]}\n{merged[1]}".strip()
            merged = merged[1:]

        if len(merged) >= 2 and self._visible_len(merged[-1]) < min_len:
            merged[-2] = f"{merged[-2]}\n{merged[-1]}".strip()
            merged = merged[:-1]

        return merged

    def _local_should_split_on_sentence_punct(self, current_text: str, delimiter: str, target_length: int) -> bool:
        visible_len = self._visible_len(current_text)
        if visible_len <= 0:
            return False
        if target_length <= 0:
            return visible_len >= 12
        return visible_len >= max(8, target_length // 2)

    def _local_split_text_core(self, text: str, apply_limits: bool) -> List[str]:
        source = str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
        if not source:
            return []
        if not self._segment_enabled():
            return [source]

        target_length = self._local_split_target_length(source)
        tokens = self._local_tokenize(source)
        if not tokens:
            return [source]

        segments: List[str] = []
        current = ""
        stack: List[str] = []
        quote_chars = {'"', "'", '`'}
        pair_map = {
            '"': '"',
            '《': '》',
            '（': '）',
            '(': ')',
            '[': ']',
            '{': '}',
            '\u2018': '\u2019',  # Chinese single quotes
            '【': '】',
        }

        def flush_current():
            nonlocal current
            seg = current.strip()
            if seg:
                segments.append(seg)
            current = ""

        for token in tokens:
            if self._local_is_atomic_token(token):
                current += token
                continue

            i = 0
            while i < len(token):
                if token.startswith("\n\n", i) and not stack:
                    flush_current()
                    i += 2
                    while i < len(token) and token[i] == "\n":
                        i += 1
                    continue

                char = token[i]
                if char in quote_chars:
                    if stack and stack[-1] == char:
                        stack.pop()
                    else:
                        stack.append(char)
                    current += char
                    i += 1
                    continue

                if stack:
                    expected = pair_map.get(stack[-1])
                    if char == expected:
                        stack.pop()
                    elif char in pair_map and char not in quote_chars:
                        stack.append(char)
                    current += char
                    i += 1
                    continue

                if char in pair_map:
                    stack.append(char)
                    current += char
                    i += 1
                    continue

                if char == "\n":
                    flush_current()
                    i += 1
                    continue

                strong_match = re.match(r"(?:[。！？?!]+|……+|…+|\.{3,})", token[i:])
                if strong_match:
                    delimiter = strong_match.group(0)
                    current += delimiter
                    if self._local_should_split_on_sentence_punct(current, delimiter, target_length):
                        flush_current()
                    i += len(delimiter)
                    continue

                current += char
                i += 1

        if current.strip():
            segments.append(current.strip())

        normalized = self._merge_local_short_segments(segments or [source], target_length)
        return self._apply_segment_limits(normalized or [source], source) if apply_limits else (normalized or [source])

    def _local_split_text(self, text: str) -> List[str]:
        return self._local_split_text_core(text, apply_limits=True)

    def _local_process_segments(self, text: str) -> List[str]:
        if not self._segment_enabled():
            return [str(text or "").strip()] if str(text or "").strip() else []
        return self._local_split_text(text)

    async def _judge_reply(
        self,
        event: AstrMessageEvent,
        reply_text: str,
        reject_reason: str = "",
    ) -> Tuple[Optional[Dict[str, Any]], float, bool]:
        providers = self._provider_candidates()
        if not providers:
            self._debug("未配置主模型，跳过处理")
            return None, 0.0, True

        if not self._any_post_process_enabled():
            return {"action": "accept", "clean_text": reply_text, "segments": [reply_text]}, 0.0, False

        values = {
            "reply_text": reply_text,
            "step_a_block": self._compose_step_a_block(),
            "step_b_block": self._compose_step_b_block(),
            "step_c_block": self._compose_step_c_block(),
            "step_d_block": self._compose_step_d_block(),
            "output_format_block": self._compose_output_format_block(),
            "judge_rule_block": self._compose_judge_rule_block(),
            "clean_rule_block": self._compose_clean_rule_block(),
            "segment_rule_block": self._compose_segment_rule_block(),
            "placeholder_rule_block": self._compose_placeholder_rule_block(reply_text),
            "reject_reason_block": self._compose_reject_reason_block(reject_reason),
        }
        prompt = self._render_template(
            str(self._cfg("judge_prompt_template", DEFAULT_JUDGE_PROMPT) or DEFAULT_JUDGE_PROMPT),
            values,
        )
        self._debug(f"审查模型输入={prompt[:3000]}")
        result_text, provider_id, elapsed, exhausted = await self._call_llm_with_fallback(prompt, stage="后处理审查")
        if exhausted:
            return None, elapsed, True
        self._debug(f"审查模型原始输出 provider={provider_id} output={result_text[:3000]}")
        parsed = self._extract_json_object(result_text)
        if not parsed:
            self._warn("审查模型未返回可解析 JSON，本次后处理已跳过")
            return None, elapsed, False
        return parsed, elapsed, False

    async def _retry_generate(
        self,
        event: AstrMessageEvent,
        reply_text: str,
        reject_reason: str,
    ) -> Tuple[str, float, bool]:
        if not self._review_enabled():
            return "", 0.0, False

        providers = self._provider_candidates()
        if not providers:
            self._debug("未配置主模型，跳过重写")
            return "", 0.0, True

        values = {
            "reply_text": reply_text,
            "reject_reason": reject_reason or "回复不适合直接发送",
            "retry_rule_block": self._compose_retry_rule_block(),
            "retry_judge_block": self._compose_judge_rule_block(),
            "retry_clean_block": self._compose_clean_rule_block(),
            "retry_segment_block": self._compose_segment_rule_block(),
            "placeholder_rule_block": self._compose_placeholder_rule_block(reply_text),
        }
        prompt = self._render_template(
            str(self._cfg("retry_prompt_template", DEFAULT_RETRY_PROMPT) or DEFAULT_RETRY_PROMPT),
            values,
        )
        self._debug(f"打回重写输入={prompt[:3000]}")
        try:
            text, provider_id, elapsed, exhausted = await self._call_llm_with_fallback(prompt, stage="打回重写")
            if exhausted:
                return "", elapsed, True
            self._debug(f"打回重写原始输出 provider={provider_id} output={text[:3000]}")
            return text, elapsed, False
        except Exception as e:
            logger.warning(f"[PostSplitter] 打回重生成失败: {e}")
            return "", 0.0, True

    async def _send_retry_notice(self, event: AstrMessageEvent):
        pool = self._cfg("retry_notice_pool", []) or []
        if not isinstance(pool, list):
            return
        candidates = [str(x).strip() for x in pool if str(x).strip()]
        if not candidates:
            return
        try:
            import random

            notice = random.choice(candidates)
            mc = MessageChain()
            mc.chain = [Plain(notice)]
            await self.context.send_message(event.unified_msg_origin, mc)
            await asyncio.sleep(0.8)
        except Exception as e:
            logger.warning(f"[PostSplitter] 发送打回提示失败: {e}")

    def _serialize_chain_for_processing(
        self, original_chain: List[BaseMessageComponent]
    ) -> Tuple[str, Dict[str, BaseMessageComponent], bool]:
        parts: List[str] = []
        placeholder_map: Dict[str, BaseMessageComponent] = {}
        comp_index = 0

        for comp in original_chain:
            if isinstance(comp, Reply):
                continue
            if isinstance(comp, Plain):
                text = comp.text or ""
                if text:
                    parts.append(text)
                continue

            comp_index += 1
            marker = f"[[RP_COMP_{comp_index}]]"
            try:
                placeholder_map[marker] = copy.deepcopy(comp)
            except Exception as e:
                self._warn(
                    f"检测到无法安全复制的富媒体组件，已跳过后处理。"
                    f" type={type(comp).__name__}, error={e}"
                )
                return "", {}, True
            parts.append(marker)

        return "".join(parts), placeholder_map, False

    def _find_original_reply_component(self, original_chain: List[BaseMessageComponent]) -> Optional[Reply]:
        for comp in original_chain:
            if isinstance(comp, Reply):
                return comp
        return None

    def _text_to_components(
        self,
        text: str,
        placeholder_map: Dict[str, BaseMessageComponent],
    ) -> List[BaseMessageComponent]:
        if not text:
            return []
        parts = re.split(f"({self.PLACEHOLDER_PATTERN.pattern})", text)
        chain: List[BaseMessageComponent] = []
        for part in parts:
            if not part:
                continue
            if self.PLACEHOLDER_PATTERN.fullmatch(part):
                comp = placeholder_map.get(part)
                if comp is not None:
                    chain.append(copy.deepcopy(comp))
                else:
                    chain.append(Plain(part))
            else:
                chain.append(Plain(part))
        return chain

    def _build_segment_chains(
        self,
        event: AstrMessageEvent,
        segments_text: List[str],
        original_reply: Optional[Reply],
        placeholder_map: Dict[str, BaseMessageComponent],
    ) -> List[List[BaseMessageComponent]]:
        chains: List[List[BaseMessageComponent]] = []
        for idx, text in enumerate(segments_text):
            chain: List[BaseMessageComponent] = []
            if idx == 0:
                if original_reply is not None:
                    chain.append(copy.deepcopy(original_reply))
                else:
                    message_obj = getattr(event, "message_obj", None)
                    if bool(self._cfg("enable_reply", True)) and getattr(message_obj, "message_id", None):
                        chain.append(Reply(id=message_obj.message_id))
            chain.extend(self._text_to_components(text, placeholder_map))
            chains.append(chain)
        return chains

    def _safe_float(self, value: Any, default: float) -> float:
        try:
            return float(value)
        except Exception:
            return default

    def _calculate_segment_delay(self, text: str) -> float:
        mode = str(self._cfg("segment_delay_mode", "linear") or "linear").strip().lower()
        max_delay = self._safe_float(self._cfg("segment_delay_max", 5.0), 5.0)
        if max_delay < 0:
            max_delay = 0.0
        if mode == "linear":
            delay_per_char = self._safe_float(self._cfg("segment_delay_per_char", 0.08), 0.08)
            delay = max(0.0, len(text or "") * delay_per_char)
            return min(delay, max_delay)
        fixed_delay = self._safe_float(self._cfg("segment_fixed_delay", 0.8), 0.8)
        delay = max(0.0, fixed_delay)
        return min(delay, max_delay)

    async def _send_segment_prefixes(self, event: AstrMessageEvent, segments: List[List[BaseMessageComponent]]):
        for idx in range(len(segments) - 1):
            segment_chain = segments[idx]
            text_content = "".join(c.text for c in segment_chain if isinstance(c, Plain))
            stripped = text_content.strip()
            has_non_plain = any(not isinstance(c, Plain) for c in segment_chain)
            if not stripped and not has_non_plain:
                continue
            mc = MessageChain()
            mc.chain = segment_chain
            await self.context.send_message(event.unified_msg_origin, mc)
            if idx < len(segments) - 2:
                delay = self._calculate_segment_delay(text_content)
                if delay > 0:
                    await asyncio.sleep(delay)

    async def _process_reply(self, event: AstrMessageEvent, original_text: str) -> Tuple[Optional[List[str]], float, str]:
        total_model_elapsed = 0.0

        judge_data, elapsed, exhausted = await self._judge_reply(event, original_text)
        total_model_elapsed += elapsed
        if exhausted:
            return self._local_process_segments(original_text), total_model_elapsed, "local_after_judge_timeout"
        if not judge_data:
            self._warn("审查模型未返回可解析 JSON，已回退本地分段")
            return self._local_process_segments(original_text), total_model_elapsed, "local_after_json_parse_failure"
        if self._should_force_local_fallback(judge_data):
            self._warn(f"审查模型命中特殊拒绝原因，已忽略模型结果并回退本地分段。reason={self._forced_local_reason()}")
            return self._local_process_segments(original_text), total_model_elapsed, "local_forced_by_reason"

        action = str(judge_data.get("action") or "accept").strip().lower()
        reason = str(judge_data.get("reason") or "").strip()
        self._debug(f"初审 action={action}, reason={reason}")

        clean_text = str(judge_data.get("clean_text") or "").strip()
        if not clean_text:
            self._warn("clean_text 为空，已 fallback 到原文")
            clean_text = original_text
            judge_data["clean_text"] = clean_text
        original_placeholders = self.PLACEHOLDER_PATTERN.findall(original_text)
        if original_placeholders:
            cleaned_placeholders = self.PLACEHOLDER_PATTERN.findall(clean_text)
            original_counter = Counter(original_placeholders)
            cleaned_counter = Counter(cleaned_placeholders)
            missing = []
            for ph, count in original_counter.items():
                missing_count = count - cleaned_counter.get(ph, 0)
                if missing_count > 0:
                    missing.extend([ph] * missing_count)
            if missing:
                judge_data["clean_text"] = clean_text + "".join(missing)
                self._warn(f"清洗后占位符丢失，已自动补回。丢失的占位符={missing}")

        first_pass_segments = self._normalize_segments(judge_data, original_text)
        first_pass_segments = self._try_restore_trailing_placeholders(original_text, first_pass_segments)
        first_pass_segments = self._final_placeholder_fallback(original_text, first_pass_segments)
        if action != "reject_and_retry":
            if self._validate_preserved_content(original_text, first_pass_segments):
                return first_pass_segments, total_model_elapsed, "accept"
            return [original_text], total_model_elapsed, "accept_but_reverted"

        if not self._review_enabled():
            if self._validate_preserved_content(original_text, first_pass_segments):
                return first_pass_segments, total_model_elapsed, "review_disabled"
            return [original_text], total_model_elapsed, "review_disabled_reverted"

        first_pass_fallback = first_pass_segments if self._validate_preserved_content(original_text, first_pass_segments) else self._local_process_segments(original_text)

        await self._send_retry_notice(event)
        regenerated, elapsed, exhausted = await self._retry_generate(event, original_text, reason)
        total_model_elapsed += elapsed
        if exhausted or not regenerated:
            base_text = "\n\n".join(first_pass_fallback).strip() or original_text
            return self._local_process_segments(base_text), total_model_elapsed, "local_after_retry_timeout"

        second_judge, elapsed, exhausted = await self._judge_reply(event, regenerated, reject_reason=reason)
        total_model_elapsed += elapsed
        if exhausted or not second_judge:
            return self._local_process_segments(regenerated), total_model_elapsed, "local_after_rejudge_timeout"
        if self._should_force_local_fallback(second_judge):
            self._warn(f"复审模型命中特殊拒绝原因，已忽略模型结果并回退本地分段。reason={self._forced_local_reason()}")
            return self._local_process_segments(regenerated), total_model_elapsed, "local_forced_by_rejudge_reason"

        second_action = str(second_judge.get("action") or "accept").strip().lower()
        second_reason = str(second_judge.get("reason") or "").strip()
        self._debug(f"复审 action={second_action}, reason={second_reason}")

        if second_action == "reject_and_retry":
            self._warn("重生成结果复审仍未通过，已回退为首轮结果/原始回复，避免发送二次判退文本")
            return first_pass_fallback or [original_text], total_model_elapsed, "rejudge_rejected"

        second_segments = self._normalize_segments(second_judge, regenerated)
        second_segments = self._try_restore_trailing_placeholders(regenerated, second_segments)
        second_segments = self._final_placeholder_fallback(regenerated, second_segments)
        if self._validate_preserved_content(regenerated, second_segments):
            return second_segments, total_model_elapsed, "rejudge_accept"

        return first_pass_fallback or [original_text], total_model_elapsed, "rejudge_reverted"

    def _log_polished_output(self, original_text: str, segments_text: List[str], process_elapsed: Optional[float] = None, mode: str = ""):
        if not segments_text:
            return
        final_text = "\n".join([s for s in segments_text if s]).strip()
        if not final_text:
            return
        self._info(f"原文本：{original_text[:500]}{"..." if len(original_text) > 500 else ""}")
        if process_elapsed is None:
            self._info(f"输出完成：{len(segments_text)} 段，总长度 {len(final_text)}")
        else:
            extra = f"，处理模式 {mode}" if mode else ""
            self._info(
                f"输出完成：{len(segments_text)} 段，总长度 {len(final_text)}，模型处理耗时 {process_elapsed:.2f}s{extra}"
            )
        self._info(f"分段结果：{segments_text}")
        self._debug(f"清洗后输出={final_text[:3000]}")

    @filter.on_decorating_result(priority=-100000000000000000)
    async def on_decorating_result(self, event: AstrMessageEvent):
        if not self._any_post_process_enabled():
            return

        result = event.get_result()
        if not result or not getattr(result, "chain", None):
            return

        if getattr(result, "__post_splitter_processed", False):
            return
        setattr(result, "__post_splitter_processed", True)

        if not bool(self._cfg("enable_group_process", True)):
            group_id = getattr(getattr(event, "message_obj", None), "group_id", None)
            if group_id:
                return

        if not self._is_model_generated_reply(event, result):
            if not bool(self._cfg("process_all_replies", False)):
                return

        skip_reason = self._check_user_input_skip(event)
        if skip_reason:
            self._info(f"用户输入匹配跳过规则：{skip_reason}，跳过处理")
            return

        original_chain = list(result.chain)
        plain_text, placeholder_map, has_unsupported = self._serialize_chain_for_processing(original_chain)
        if has_unsupported:
            self._debug("检测到不支持的富媒体组件，跳过处理以避免组件位置错位")
            return
        if not plain_text.strip():
            return

        if not self._strip_placeholders(plain_text).strip():
            self._info("回复内容仅含富媒体占位符，跳过处理")
            return

        if bool(self._cfg("enable_max_process_length", True)):
            max_len = self._safe_int(self._cfg("max_process_length", 500) or 500, 500)
            if max_len > 0:
                visible_len = len(self._strip_placeholders(plain_text).strip())
                if visible_len > max_len:
                    self._info(f"文本长度 {visible_len} 超过限制 {max_len}，跳过后处理")
                    return

        original_reply = self._find_original_reply_component(original_chain)

        try:
            async with self._session_guard(event):
                segments_text, process_elapsed, process_mode = await self._process_reply(event, plain_text)

                if not segments_text:
                    return

                if not self._segment_enabled() and segments_text:
                    joined = "\n".join(seg for seg in segments_text if seg).strip() or plain_text
                    segments_text = [joined]

                if bool(self._cfg("strip_segment_trailing_period", True)):
                    segments_text = [self._strip_single_trailing_period(seg) for seg in segments_text]

                self._log_polished_output(plain_text, segments_text, process_elapsed, process_mode)

                segments = self._build_segment_chains(event, segments_text, original_reply, placeholder_map)
                if not segments:
                    return

                if len(segments) == 1:
                    result.chain.clear()
                    result.chain.extend(segments[0])
                    self._debug(f"单段结果：{segments_text[0][:300]}")
                    return

                try:
                    await self._send_segment_prefixes(event, segments)
                except Exception as e:
                    logger.error(f"[PostSplitter] 主动发送前置分段失败: {e}", exc_info=True)
                    result.chain.clear()
                    result.chain.extend(original_chain)
                    return

                result.chain.clear()
                result.chain.extend(segments[-1])
                self._debug(f"最终分为 {len(segments)} 段")
        except Exception as e:
            logger.error(f"[PostSplitter] 审查/重写流程失败: {e}", exc_info=True)
            result.chain.clear()
            result.chain.extend(original_chain)
            return
