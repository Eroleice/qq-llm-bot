from __future__ import annotations

import re

from qq_llm_bot.final_qa_models import (
    FINAL_QA_CATEGORIES as FINAL_QA_CATEGORIES,
    FinalQAResult as FinalQAResult,
    final_qa_category_for_reason as final_qa_category_for_reason,
    safe_final_qa_categories as safe_final_qa_categories,
)
from qq_llm_bot.models import ConversationSnapshot, MessageContext, PerceptionResult


UNSAFE_SELF_PATTERN = re.compile(
    r"(住在|地址|手机号|身份证|真实姓名|学校|高中|大学|公司|上班|工作在|毕业于|"
    r"爸爸|妈妈|父母|亲戚|男朋友|女朋友|老公|老婆|线下见|见面|现实里)"
)
POLITICAL_TOPIC_PATTERN = re.compile(
    r"(政治|政党|政府|国家领导|领导人|总统|主席|选举|民主|专制|独裁|革命|政权|"
    r"外交|制裁|战争|领土|台湾|香港|新疆|西藏|中共|共产党|国民党|乌克兰|"
    r"俄罗斯|以色列|巴勒斯坦)"
)
POLITICAL_STANCE_PATTERN = re.compile(
    r"(支持|反对|赞成|不赞成|站队|表态|立场|应该|不应该|必须|活该|打得好|"
    r"正义|邪恶|赢了|输了|好事|坏事|同意|不同意|确实|有道理)"
)
POLITICAL_DEFLECTION_PATTERN = re.compile(
    r"(不聊|先别聊|不站队|不表态|不评价|换个话题|这个话题不适合|别在群里聊)"
)
SYSTEM_LEAK_PATTERN = re.compile(
    r"(system prompt|系统提示|开发者消息|developer message|隐藏指令|内部提示|"
    r"api[_ -]?key|access[_ -]?token|sk-[A-Za-z0-9_-]{20,})",
    re.I,
)
PRIVACY_LEAK_PATTERN = re.compile(
    r"(\b1[3-9]\d{9}\b|\b\d{17}[\dXx]\b|身份证号|家庭住址|银行卡号|密码是)"
)
INAPPROPRIATE_REPLY_PATTERN = re.compile(
    r"(去死|自杀|杀了|狠狠干|裸照|色情|约炮|开盒|人肉|诈骗|洗钱|炸药|毒品)"
)
SHARED_CONTENT_CUE_PATTERN = re.compile(
    r"(截图|图片|图里|图上|新闻|帖子|网传|热搜|自媒体|视频|爆料|瓜|通报|公告|聊天记录|转发|链接|微博|公众号)"
)
TRUTH_VERIFICATION_REQUEST_PATTERN = re.compile(
    r"(真的假的|真(?:的)?吗|假的吗|是不是假的|靠谱吗|可信|来源呢|有来源|求证|核实|查证|验证|辟谣|谣言|"
    r"P图|p图|p的|P的|官宣|官方确认|真实吗|真实性|怎么确认|能查一下|查一下|帮.*看.*真假)"
)
UNSOLICITED_TRUTH_DOUBT_PATTERN = re.compile(
    r"(真实性存疑|真假(?:还)?不好说|不好判断真假|无法确认真假|不能确认真假|来源不明|没有来源|缺少来源|"
    r"等官宣|等官方|等官方确认|先别信|别太当真|像(?:是)?P图|像(?:是)?p图|P的|p的|谣言|"
    r"网传不一定|未经证实|未证实|不排除是假的|可能是假的|不一定是真的|有待核实|需要核实)"
)
CLEAR_FALSEHOOD_EVIDENCE_PATTERN = re.compile(
    r"(已经辟谣|官方辟谣|证实是假的|确认是假的|实锤是假的|P图实锤|p图实锤|造谣|假的|不是真的|澄清了)"
)
HIGH_RISK_SHARED_CONTENT_PATTERN = re.compile(
    r"(转账|打钱|付款|收款码|投资|理财|借钱|诈骗|验证码|密码|银行卡|身份证|手机号|住址|"
    r"医疗|吃药|用药|诊断|急救|违法|毒品|炸药|自杀|开盒|人肉)"
)
LIVE_EVENT_TOPIC_PATTERN = re.compile(
    r"(比赛|球赛|赛况|比分|直播|现场|半场|终场|加时|补时|进球|得分|领先|落后|追平|反超|绝杀|"
    r"开赛|开打|还剩|暂停|红牌|黄牌|点球|VAR|三杀|团战|大龙|小龙|推塔|对局|决赛|半决赛|"
    r"世界杯|欧冠|NBA|CBA|LPL|KPL|电竞|足球|篮球|网球|排球|棒球|乒乓|斯诺克|F1|"
    r"第[一二三四五六七八九十\d]+(?:局|节|盘|轮|场|回合)|BO\d+)",
    re.I,
)
LIVE_EVENT_TIME_PATTERN = re.compile(
    r"(现在|正在|刚刚|刚才|这会儿|此刻|实时|直播|现场|马上|还在|开打|开赛|比分|赛况|几比几|"
    r"进球|得分|领先|落后|追平|反超|绝杀|终场|半场|加时|还剩|第[一二三四五六七八九十\d]+(?:局|节|盘|轮|场|回合))"
)
LIVE_EVENT_SCORE_PATTERN = re.compile(r"(?<!\d)(\d{1,3})\s*(?:[:：-]|比)\s*(\d{1,3})(?!\d)")
LIVE_EVENT_CLAIM_PATTERN = re.compile(
    r"((?:现在|目前|刚刚|刚才|已经|这会儿|此刻).{0,18}"
    r"(?:比分|进球|得分|领先|落后|追平|反超|赢了|输了|获胜|淘汰|晋级|结束|终场|半场|加时|还剩|第[一二三四五六七八九十\d]+(?:局|节|盘|轮|场|回合)))|"
    r"((?:比分|赛况).{0,12}(?:是|变成|到了)?\s*\d{1,3}\s*(?:[:：-]|比)\s*\d{1,3})"
)
LIVE_EVENT_LIMITATION_PATTERN = re.compile(
    r"(没有实时|看不到实时|没法实时|不能实时|无法实时|没有最新|不知道当前|不清楚当前|"
    r"按你们刚才|按群里|你们刚才|群里刚才|补一下比分|发下比分|不编|别编)"
)
REPLY_HARD_TRIM_BOUNDARY_RE = re.compile(r"[。！？!?；;]")
REPLY_SOFT_TRIM_BOUNDARY_RE = re.compile(r"[，,、]")
INCOMPLETE_REPLY_END_RE = re.compile(
    r"(?:[，,、：:；;\s]+|"
    r"(?:挺|很|更|先|把|被|在|里|但|不过|然后|而且|因为|所以|如果|虽然|或者|和|跟|与|用|当|靠|像))$"
)


def contextual_final_qa_block_reason(
    context: MessageContext,
    snapshot: ConversationSnapshot,
    reply: str,
) -> str:
    truth_reason = _unsolicited_truth_doubt_reason(context, snapshot, reply)
    if truth_reason:
        return truth_reason
    live_reason = _ungrounded_live_event_claim_reason(context, snapshot, reply)
    if live_reason:
        return live_reason
    return ""


def hard_final_qa_block_reason(reply: str) -> str:
    if SYSTEM_LEAK_PATTERN.search(reply):
        return "system_leak"
    if PRIVACY_LEAK_PATTERN.search(reply):
        return "privacy"
    return ""


def heuristic_final_qa_block_reason(
    context: MessageContext,
    snapshot: ConversationSnapshot,
    reply: str,
) -> str:
    if INAPPROPRIATE_REPLY_PATTERN.search(reply):
        return "inappropriate"
    if UNSAFE_SELF_PATTERN.search(reply):
        return "unsafe_self_claim"
    scene = "\n".join([*snapshot.recent_messages[-12:], context.plain_text, reply])
    if _looks_like_political_stance(scene, reply):
        return "political_stance"
    return ""


def looks_like_live_event_context(
    context: MessageContext,
    perception: PerceptionResult,
    snapshot: ConversationSnapshot,
) -> bool:
    current = " ".join([context.plain_text, *perception.topics])
    recent = "\n".join(snapshot.recent_messages[-6:])
    scene = f"{recent}\n{current}"
    if LIVE_EVENT_SCORE_PATTERN.search(current):
        return True
    if LIVE_EVENT_TOPIC_PATTERN.search(current) and LIVE_EVENT_TIME_PATTERN.search(current):
        return True
    if LIVE_EVENT_TIME_PATTERN.search(current) and LIVE_EVENT_TOPIC_PATTERN.search(scene):
        return True
    if LIVE_EVENT_TOPIC_PATTERN.search(current) and LIVE_EVENT_TIME_PATTERN.search(scene):
        return True
    return False


def sanitize_reply(reply: str, max_chars: int) -> str:
    text = reply.strip()
    text = re.sub(r"^回复[:：]\s*", "", text)
    text = text.replace("作为AI", "").replace("作为一个AI", "")
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    return _trim_reply_to_max_chars(text, max_chars)


def _chat_scene_text(context: MessageContext, snapshot: ConversationSnapshot) -> str:
    return "\n".join(
        [
            *snapshot.recent_messages[-12:],
            *snapshot.recent_image_descriptions[-8:],
            context.plain_text,
        ]
    )


def _unsolicited_truth_doubt_reason(
    context: MessageContext,
    snapshot: ConversationSnapshot,
    reply: str,
) -> str:
    if not UNSOLICITED_TRUTH_DOUBT_PATTERN.search(reply):
        return ""
    scene = _chat_scene_text(context, snapshot)
    if not _looks_like_shared_content_scene(scene):
        return ""
    if TRUTH_VERIFICATION_REQUEST_PATTERN.search(scene):
        return ""
    if CLEAR_FALSEHOOD_EVIDENCE_PATTERN.search(scene):
        return ""
    if HIGH_RISK_SHARED_CONTENT_PATTERN.search(scene):
        return ""
    return "unsolicited_truth_doubt"


def _looks_like_shared_content_scene(scene: str) -> bool:
    return bool(
        SHARED_CONTENT_CUE_PATTERN.search(scene)
        or "[图片解读]" in scene
        or "[图片文字]" in scene
        or "[图片]" in scene
    )


def _ungrounded_live_event_claim_reason(
    context: MessageContext,
    snapshot: ConversationSnapshot,
    reply: str,
) -> str:
    if LIVE_EVENT_LIMITATION_PATTERN.search(reply):
        return ""
    scene = _chat_scene_text(context, snapshot)
    combined = f"{scene}\n{reply}"
    if not (LIVE_EVENT_TOPIC_PATTERN.search(combined) and LIVE_EVENT_TIME_PATTERN.search(combined)):
        return ""

    reply_scores = _normalized_live_scores(reply)
    if reply_scores:
        scene_scores = _normalized_live_scores(scene)
        if any(score not in scene_scores for score in reply_scores):
            return "ungrounded_live_event_claim"

    claim = LIVE_EVENT_CLAIM_PATTERN.search(reply)
    if claim and not _live_claim_has_scene_evidence(claim.group(0), scene):
        return "ungrounded_live_event_claim"
    return ""


def _normalized_live_scores(text: str) -> set[str]:
    scores: set[str] = set()
    for match in LIVE_EVENT_SCORE_PATTERN.finditer(text):
        left, right = match.groups()
        scores.add(f"{int(left)}:{int(right)}")
    return scores


def _live_claim_has_scene_evidence(claim: str, scene: str) -> bool:
    if _normalized_live_scores(claim) & _normalized_live_scores(scene):
        return True
    result_terms = (
        "进球",
        "得分",
        "领先",
        "落后",
        "追平",
        "反超",
        "赢了",
        "输了",
        "获胜",
        "淘汰",
        "晋级",
        "结束",
        "终场",
        "半场",
        "加时",
        "还剩",
    )
    if any(term in claim and term in scene for term in result_terms):
        return True
    round_match = re.search(r"第[一二三四五六七八九十\d]+(?:局|节|盘|轮|场|回合)", claim)
    return bool(round_match and round_match.group(0) in scene)


def _looks_like_political_stance(scene: str, reply: str) -> bool:
    if POLITICAL_DEFLECTION_PATTERN.search(reply):
        return False
    reply_has_topic = bool(POLITICAL_TOPIC_PATTERN.search(reply))
    scene_has_topic = bool(POLITICAL_TOPIC_PATTERN.search(scene))
    if not reply_has_topic and not scene_has_topic:
        return False
    if POLITICAL_STANCE_PATTERN.search(reply):
        return True
    if reply_has_topic and re.search(r"(怎么看|谁对|评价|立场|观点|新闻|冲突|战争|选举)", scene):
        return True
    return False


def _trim_reply_to_max_chars(text: str, max_chars: int) -> str:
    clipped = text[:max_chars].strip()
    min_reasonable_length = min(12, max(1, max_chars // 3))
    terminal = _last_boundary_match(clipped, REPLY_HARD_TRIM_BOUNDARY_RE)
    if terminal is not None and terminal.end() >= min_reasonable_length:
        return clipped[: terminal.end()].strip()

    soft = _last_boundary_match(clipped, REPLY_SOFT_TRIM_BOUNDARY_RE)
    if soft is not None and soft.start() >= min_reasonable_length:
        return _strip_incomplete_reply_tail(clipped[: soft.start()])

    trimmed = _strip_incomplete_reply_tail(clipped)
    return trimmed or clipped


def _last_boundary_match(text: str, pattern: re.Pattern[str]) -> re.Match[str] | None:
    matches = list(pattern.finditer(text))
    return matches[-1] if matches else None


def _strip_incomplete_reply_tail(text: str) -> str:
    cleaned = text.strip()
    while cleaned:
        next_cleaned = INCOMPLETE_REPLY_END_RE.sub("", cleaned).strip()
        if next_cleaned == cleaned:
            break
        cleaned = next_cleaned
    return cleaned
