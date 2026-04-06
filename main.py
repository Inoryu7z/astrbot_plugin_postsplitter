import asyncio
import copy
import json
import re
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Tuple

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, MessageChain, filter
from astrbot.api.message_components import BaseMessageComponent, Plain, Reply
from astrbot.api.provider import LLMResponse
from astrbot.api.star import Context, Star, register


DEFAULT_FORCE_LOCAL_REASON = "内容涉及不适合主题，拒绝处理"

DEFAULT_JUDGE_PROMPT = """# 角色
你是聊天回复后处理器，只负责在不改变原意的前提下，对候选回复进行必要处理。

# 已知上下文
- 用户消息：{{user_message}}
- 候选回复：{{reply_text}}

# 处理要求
1. 如果候选回复属于不适合交给你继续后处理的主题类别，则不要执行清洗、改写、重写或主观审查式分段判断，并令 reason = "内容涉及不适合主题，拒绝处理"。
2. 命中上述情况时，此要求优先于其他后处理规则；同时应保守返回，不得改写原意，不得删改事实内容。

{{judge_rule_block}}
{{clean_rule_block}}
{{segment_rule_block}}
{{retry_rule_block}}
{{placeholder_rule_block}}

## 输出格式
{{output_format_block}}

## 严格禁止
- 不得歪曲原意
- 不得偷换含义
- 不得弱化、强化、扭曲原文情绪与立场
- 不得补写原文没有的事实、设定、关系、动机、经历
- 不得为了清洗、判别或分段而删除关键信息
- 不得为了人设合规而把原文改成另一种意思
- 不得新增原文不存在的人称、关系、设定、时间、地点、金额、数字、结论、承诺、经历
- 不得删除或改写原文中的数字、URL、代码、引用片段、专有名词、时间信息
- 未启用的功能，不得自行执行对应处理
- 如果无法在当前已启用功能范围内安全处理，必须保守输出；仅在已启用判别与打回时，才允许返回 `reject_and_retry`
- 不要输出 JSON 以外内容
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

{{retry_judge_block}}
{{retry_clean_block}}
{{retry_segment_block}}
{{placeholder_rule_block}}

## 上下文
- 用户刚刚的话：{{user_message}}
- 上一版候选回复：{{reply_text}}
- 打回原因：{{reject_reason}}
"""


@register(
    "astrbot_plugin_postsplitter",
    "Inoryu7z",
    "基于 LLM 的回复后处理分段器：优先对回复做自然分段，并支持自定义清洗、审查与打回重生成。",
    "1.2.4",
)
class PostSplitterPlugin(Star):
    URL_PATTERN = re.compile(r"https?://[^\s]+", re.IGNORECASE)
    CODE_FENCE_PATTERN = re.compile(r"```[\s\S]*?```")
    NUMBER_PATTERN = re.compile(r"\d+(?:[./:\-]\d+)*")
    MENTION_PATTERN = re.compile(r"(?<![\w@])@[A-Za-z0-9_\-\u4e00-\u9fff]+")
    PLACEHOLDER_PATTERN = re.compile(r"\[\[RP_COMP_\d+\]\]")
    TRAILING_PLACEHOLDERS_PATTERN = re.compile(r"(?P<trailing>(?:\[\[RP_COMP_\d+\]\])+)+\s*$")
    LOCAL_SPLIT_PATTERN = re.compile(r"([。！？?!；;\n]+)")

    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        self._session_locks: Dict[str, asyncio.Lock] = {}
        self._session_lock_refs: Dict[str, int] = {}

    @filter.on_llm_response()
    async def on_llm_response(self, event: AstrMessageEvent, resp: LLMResponse):
        setattr(event, "__reply_polisher_is_llm_reply", True)

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
            logger.info(f"[ReplyPolisher] {message}")

    def _info(self, message: str):
        logger.info(f"[ReplyPolisher] {message}")

    def _warn(self, message: str):
        logger.warning(f"[ReplyPolisher] {message}")

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
        return str(self._cfg("force_local_fallback_reason", DEFAULT_FORCE_LOCAL_REASON) or DEFAULT_FORCE_LOCAL_REASON).strip()

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
                logger.debug(f"[ReplyPolisher] is_model_result() 判定失败，尝试回退: {e}")

        is_llm_result = getattr(result, "is_llm_result", None)
        if callable(is_llm_result):
            try:
                if is_llm_result():
                    return True
            except Exception as e:
                logger.debug(f"[ReplyPolisher] is_llm_result() 判定失败，尝试回退: {e}")

        content_type = getattr(result, "result_content_type", None)
        if content_type is not None:
            type_name = getattr(content_type, "name", "")
            return type_name in {"LLM_RESULT", "AGENT_RUNNER_RESULT"}

        return getattr(event, "__reply_polisher_is_llm_reply", False)

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
        if "\n\n" in source:
            return [seg.strip() for seg in source.split("\n\n") if seg.strip()]
        return [source.strip()]

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
                head = normalized[: max_seg - 1] if max_seg > 1 else []
                tail = normalized[max_seg - 1 :]
                merged_tail = "\n\n".join(seg for seg in tail if seg).strip()
                normalized = head + ([merged_tail] if merged_tail else [])
            if len(normalized) < min_seg:
                source = "\n\n".join(normalized).strip() or str(fallback_text or "").strip()
                normalized = self._build_fallback_segments_from_text(source)
        else:
            if len(normalized) > 12:
                head = normalized[:11]
                tail = normalized[11:]
                merged_tail = "\n\n".join(seg for seg in tail if seg).strip()
                normalized = head + ([merged_tail] if merged_tail else [])

        return normalized

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
                        self._warn("segments 与 clean_text 存在占位符差异，已自动回填占位符并保留分段")
                        return self._apply_segment_limits(reinjected, clean_text or fallback_text)
                    self._warn("segments 与 clean_text 存在可见差异，已优先信任 clean_text 并回退为本地单段")
                    return self._build_fallback_segments_from_text(clean_text)
            return self._apply_segment_limits(normalized, clean_text or fallback_text)

        if clean_text:
            return self._build_fallback_segments_from_text(clean_text)

        fallback_text = str(fallback_text or "").strip()
        return [fallback_text] if fallback_text else []

    def _normalize_url_token(self, token: str) -> str:
        return (token or "").strip().rstrip("'\"）)]}，。！？；：,.!?;:")

    def _normalize_code_token(self, token: str) -> str:
        code = (token or "").replace("\r\n", "\n").replace("\r", "\n")
        lines = [line.rstrip() for line in code.split("\n")]
        return "\n".join(lines).strip()

    def _normalize_number_token(self, token: str) -> str:
        return re.sub(r"\D", "", token or "")

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
        return "分段时，优先模拟真人在即时聊天中的发送习惯。允许把短促起手句、语气过渡句、临时停顿句单独成段，例如“我去”“等等”“不是”。相比段长均匀，更优先保留自然聊天节奏。"

    def _compose_output_format_block(self) -> str:
        if self._segment_enabled():
            return (
                "只能输出 JSON，不要输出解释、前后缀、Markdown：\n\n"
                "{\n"
                '  "action": "accept 或 reject_and_retry",\n'
                '  "reason_type": "normal|system_error|persona_mismatch|weird_text|other",\n'
                f'  "reason": "简短原因；若不适合继续后处理则固定为 {self._forced_local_reason()}",\n'
                '  "clean_text": "清洗后的完整文本",\n'
                '  "segments": ["分段1", "分段2"],\n'
                '  "confidence": 0.0\n'
                "}"
            )
        return (
            "只能输出 JSON，不要输出解释、前后缀、Markdown：\n\n"
            "{\n"
            '  "action": "accept 或 reject_and_retry",\n'
            '  "reason_type": "normal|system_error|persona_mismatch|weird_text|other",\n'
            f'  "reason": "简短原因；若不适合继续后处理则固定为 {self._forced_local_reason()}",\n'
            '  "clean_text": "清洗后的完整文本",\n'
            '  "confidence": 0.0\n'
            "}"
        )

    def _compose_placeholder_rule_block(self, reply_text: str) -> str:
        placeholders = self.PLACEHOLDER_PATTERN.findall(reply_text or "")
        if not placeholders:
            return ""
        return (
            "## 占位符保留规则\n"
            "正文中可能包含形如 [[RP_COMP_数字]] 的组件占位符。它们代表原消息中的非文本内联组件。\n"
            "这些占位符必须原样保留，不得删除、改写、翻译、拆开、合并、补空格或调整顺序。"
        )

    def _compose_judge_rule_block(self) -> str:
        if not self._review_enabled():
            return ""
        judge_prompt_input = str(self._cfg("persona_style_rules", "") or "").strip()
        return (
            "## 判别规则\n"
            "需要根据下列额外事项判断回复是否合规。若存在严重违背，则返回 `reject_and_retry`。\n\n"
            f"{judge_prompt_input}"
        ).strip()

    def _compose_clean_rule_block(self) -> str:
        if not self._clean_enabled():
            return ""
        clean_prompt_input = str(self._cfg("clean_prompt_template", "") or "").strip()
        return (
            "## 清洗规则\n"
            "清洗的目标，不只是做表面整理，而是在不改变原意、不新增事实、不删减关键信息的前提下，把不适合直接发送的表达修正为适合直接发送的表达。\n"
            "允许处理的方向包括：格式整理、符号修正、空行压缩、错别字修正、异体字与繁简统一、非术语英文清洗，以及对明显别扭、失衡、过脏、过乱、过于不适合直接发送的表述做保守修整。\n"
            "但这种修整仅限于让表达更可直接发送，不得改变原意，不得偷换语气立场，不得新增、删减、编造事实内容。\n\n"
            f"{clean_prompt_input}"
        ).strip()

    def _compose_segment_rule_block(self) -> str:
        if not self._segment_enabled():
            return ""
        parts = [
            "## 分段规则",
            "需要处理分段。请根据语义决定是否分段，并输出最终 `segments`。",
        ]
        count_rule = self._build_segment_count_rule_text().strip()
        pref_rule = self._build_segment_preference_rule_text().strip()
        if count_rule:
            parts.append(count_rule)
        if pref_rule:
            parts.append(pref_rule)
        return "\n\n".join(parts).strip()

    def _compose_retry_rule_block(self) -> str:
        if not self._review_enabled():
            return ""
        return (
            "## 打回规则\n"
            "当问题严重且无法在不改变原意的前提下直接修复时，返回 `reject_and_retry`。重写时仍必须保持原意，不得借机二次创作。"
        )

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

    def _split_large_local_segment(self, text: str, target_length: int) -> List[str]:
        source = str(text or "").strip()
        if not source:
            return []
        if target_length <= 0 or len(self._strip_placeholders(source)) <= max(target_length * 2, 48):
            return [source]

        pieces = re.split(r"([，,、：:；;])", source)
        if len(pieces) <= 1:
            return [source]

        segments: List[str] = []
        current = ""
        for piece in pieces:
            if not piece:
                continue
            candidate = f"{current}{piece}"
            visible_len = len(self._strip_placeholders(candidate).strip())
            if current and visible_len >= target_length:
                segments.append(current.strip())
                current = piece
            else:
                current = candidate
        if current.strip():
            segments.append(current.strip())

        normalized = [seg for seg in segments if seg]
        return normalized or [source]

    def _merge_local_short_segments(self, segments: List[str], target_length: int) -> List[str]:
        normalized = [str(seg or "").strip() for seg in segments if str(seg or "").strip()]
        if not normalized:
            return []

        min_len = 6 if target_length <= 0 else max(6, min(12, target_length // 3))
        merged: List[str] = []
        for seg in normalized:
            visible_len = len(self._strip_placeholders(seg).strip())
            if merged and visible_len < min_len:
                connector = "" if re.match(r"^[。！？?!；;，,、：:]", seg) else "\n"
                merged[-1] = f"{merged[-1]}{connector}{seg}".strip()
            else:
                merged.append(seg)

        if len(merged) >= 2 and len(self._strip_placeholders(merged[0]).strip()) < min_len:
            merged[1] = f"{merged[0]}\n{merged[1]}".strip()
            merged = merged[1:]

        return merged

    def _local_split_text(self, text: str) -> List[str]:
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
            '“': '”',
            '《': '》',
            '（': '）',
            '(': ')',
            '[': ']',
            '{': '}',
            '‘': '’',
            '【': '】',
            '<': '>',
        }

        def flush_current():
            nonlocal current
            seg = current.strip()
            if seg:
                segments.append(seg)
            current = ""

        def append_and_maybe_split(delimiter: str):
            nonlocal current
            current += delimiter
            visible_len = len(self._strip_placeholders(current).strip())
            if delimiter.count("\n") >= 2:
                flush_current()
                return
            if re.fullmatch(r"[。！？?!；;]+", delimiter or ""):
                if target_length <= 0 or visible_len >= max(8, target_length // 2):
                    flush_current()
                return
            if "\n" in delimiter and visible_len >= max(12, target_length or 18):
                flush_current()

        for token in tokens:
            if self._local_is_atomic_token(token):
                current += token
                continue

            i = 0
            while i < len(token):
                if token.startswith("\n\n", i) and not stack:
                    append_and_maybe_split("\n\n")
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

                strong_match = re.match(r"[。！？?!；;]+", token[i:])
                if strong_match:
                    delimiter = strong_match.group(0)
                    append_and_maybe_split(delimiter)
                    i += len(delimiter)
                    continue

                if char == "\n":
                    append_and_maybe_split("\n")
                    i += 1
                    continue

                current += char
                i += 1

        if current.strip():
            segments.append(current.strip())

        if not segments:
            segments = [source]

        normalized: List[str] = []
        for seg in segments:
            normalized.extend(self._split_large_local_segment(seg, target_length))

        normalized = self._merge_local_short_segments(normalized, target_length)
        return self._apply_segment_limits(normalized or [source], source)

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
            "user_message": getattr(event, "message_str", "") or "",
            "judge_rule_block": self._compose_judge_rule_block(),
            "clean_rule_block": self._compose_clean_rule_block(),
            "segment_rule_block": self._compose_segment_rule_block(),
            "retry_rule_block": self._compose_retry_rule_block(),
            "placeholder_rule_block": self._compose_placeholder_rule_block(reply_text),
            "output_format_block": self._compose_output_format_block(),
            "reject_reason": reject_reason,
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
            "user_message": getattr(event, "message_str", "") or "",
            "reject_reason": reject_reason or "回复不适合直接发送",
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
            logger.warning(f"[ReplyPolisher] 打回重生成失败: {e}")
            return "", 0.0, True

    async def _send_retry_notice(self, event: AstrMessageEvent):
        if not bool(self._cfg("enable_retry_notice", False)):
            return
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
            logger.warning(f"[ReplyPolisher] 发送打回提示失败: {e}")

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
                elif bool(self._cfg("enable_reply", True)) and getattr(event.message_obj, "message_id", None):
                    chain.append(Reply(id=event.message_obj.message_id))
            chain.extend(self._text_to_components(text, placeholder_map))
            chains.append(chain)
        return chains

    def _safe_float(self, value: Any, default: float) -> float:
        try:
            return float(value)
        except Exception:
            return default

    def _calculate_segment_delay(self, text: str) -> float:
        mode = str(self._cfg("segment_delay_mode", "fixed") or "fixed").strip().lower()
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
            return None, total_model_elapsed, "skip"
        if self._should_force_local_fallback(judge_data):
            self._warn(f"审查模型命中特殊拒绝原因，已忽略模型结果并回退本地分段。reason={self._forced_local_reason()}")
            return self._local_process_segments(original_text), total_model_elapsed, "local_forced_by_reason"

        action = str(judge_data.get("action") or "accept").strip().lower()
        reason = str(judge_data.get("reason") or "").strip()
        self._debug(f"初审 action={action}, reason={reason}")

        first_pass_segments = self._normalize_segments(judge_data, original_text)
        first_pass_segments = self._try_restore_trailing_placeholders(original_text, first_pass_segments)
        if action != "reject_and_retry":
            if self._validate_preserved_content(original_text, first_pass_segments):
                return first_pass_segments, total_model_elapsed, "accept"
            return [original_text], total_model_elapsed, "accept_but_reverted"

        if not self._review_enabled():
            if self._validate_preserved_content(original_text, first_pass_segments):
                return first_pass_segments, total_model_elapsed, "review_disabled"
            return [original_text], total_model_elapsed, "review_disabled_reverted"

        first_pass_fallback = first_pass_segments if self._validate_preserved_content(original_text, first_pass_segments) else [original_text]

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
        second_segments = self._try_restore_trailing_placeholders(original_text, second_segments)
        if self._validate_preserved_content(original_text, second_segments):
            return second_segments, total_model_elapsed, "rejudge_accept"

        return first_pass_fallback or [original_text], total_model_elapsed, "rejudge_reverted"

    def _log_polished_output(self, segments_text: List[str], process_elapsed: Optional[float] = None, mode: str = ""):
        if not segments_text:
            return
        final_text = "\n".join([s for s in segments_text if s]).strip()
        if not final_text:
            return
        if process_elapsed is None:
            self._info(f"输出完成：{len(segments_text)} 段，总长度 {len(final_text)}")
        else:
            extra = f"，处理模式 {mode}" if mode else ""
            self._info(
                f"输出完成：{len(segments_text)} 段，总长度 {len(final_text)}，模型处理耗时 {process_elapsed:.2f}s{extra}"
            )
        self._debug(f"清洗后输出={final_text[:3000]}")

    @filter.on_decorating_result(priority=-100000000000000000)
    async def on_decorating_result(self, event: AstrMessageEvent):
        if not self._any_post_process_enabled():
            return

        result = event.get_result()
        if not result or not getattr(result, "chain", None):
            return

        if getattr(result, "__reply_polisher_processed", False):
            return
        setattr(result, "__reply_polisher_processed", True)

        if not bool(self._cfg("enable_group_process", True)):
            group_id = getattr(getattr(event, "message_obj", None), "group_id", None)
            if group_id:
                return

        if not self._is_model_generated_reply(event, result):
            return

        original_chain = list(result.chain)
        plain_text, placeholder_map, has_unsupported = self._serialize_chain_for_processing(original_chain)
        if has_unsupported:
            self._debug("检测到不支持的富媒体组件，跳过处理以避免组件位置错位")
            return
        if not plain_text.strip():
            return

        original_reply = self._find_original_reply_component(original_chain)

        try:
            async with self._session_guard(event):
                segments_text, process_elapsed, process_mode = await self._process_reply(event, plain_text)

                if not segments_text:
                    if bool(self._cfg("fallback_to_original_on_error", True)):
                        return
                    segments_text = [plain_text]

                if not self._segment_enabled() and segments_text:
                    joined = "\n".join(seg for seg in segments_text if seg).strip() or plain_text
                    segments_text = [joined]

                self._log_polished_output(segments_text, process_elapsed, process_mode)

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
                    logger.error(f"[ReplyPolisher] 主动发送前置分段失败: {e}", exc_info=True)
                    if bool(self._cfg("fallback_to_original_on_error", True)):
                        result.chain.clear()
                        result.chain.extend(original_chain)
                        return

                result.chain.clear()
                result.chain.extend(segments[-1])
                self._debug(f"最终分为 {len(segments)} 段")
        except Exception as e:
            logger.error(f"[ReplyPolisher] 审查/重写流程失败: {e}", exc_info=True)
            if bool(self._cfg("fallback_to_original_on_error", True)):
                return
            result.chain.clear()
            result.chain.extend(original_chain)
