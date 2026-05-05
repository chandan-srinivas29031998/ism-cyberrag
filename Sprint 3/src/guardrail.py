import re
from src.config import OOS_PRE_FILTER_ENABLED, OOS_RERANK_THRESHOLD


OOS_REFUSAL = (
    "I don't have enough information from the ISM documents to answer this. "
    "This question is outside the scope of the Australian Information Security Manual (ISM)."
)

DENY_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        # Food and cooking
        r"\brecipes?\b",
        r"\bcooking\b",
        r"\bingredients?\b",
        r"\bcalories?\b",
        r"\bnutrition\b",
        r"\bmeal plan\b",
        # Finance and markets
        r"\bstock (price|market|trading)\b",
        r"\bcryptocurrency\b",
        r"\bbitcoin\b",
        r"\binvest(ment|ing)\b",
        r"\bforex\b",
        # Entertainment and media
        r"\bmovie review\b",
        r"\bsong lyrics\b",
        r"\bcelebrit(y|ies)\b",
        r"\btv show\b",
        r"\bnetflix\b",
        r"\bspotify\b",
        r"\bbook review\b",
        r"\bgaming\b",
        r"\bvideo game\b",
        # Sports
        r"\bsports? score\b",
        r"\bnba\b",
        r"\bnfl\b",
        r"\bfifa\b",
        r"\bworld cup\b",
        r"\bolympic\b",
        r"\bpremier league\b",
        # Creative writing
        r"\bwrite me a (poem|story|essay|song|letter)\b",
        r"\bwrite a (poem|story|essay|song|letter)\b",
        r"\btell me a joke\b",
        r"\bjoke\b",
        r"\bfunny\b",
        # Personal and lifestyle
        r"\bhoroscope\b",
        r"\bastrology\b",
        r"\blottery\b",
        r"\bdating (tips|advice|app)\b",
        r"\brelationship advice\b",
        r"\bdiet (plan|tips)\b",
        r"\bworkout (plan|routine)\b",
        r"\bfitness\b",
        r"\bweight loss\b",
        r"\bfashion\b",
        r"\bmakeup\b",
        r"\bbeauty\b",
        # Travel and geography
        r"\btravel (itinerary|guide|tips)\b",
        r"\bflight (booking|deal)\b",
        r"\bhotel (booking|review)\b",
        r"\btourist\b",
        r"\bvacation\b",
        # Weather
        r"\bweather (forecast|today|tomorrow)\b",
        r"\btemperature in\b",
        # Education unrelated
        r"\bhomework help\b",
        r"\bmath problem\b",
        r"\bhistory of (rome|egypt|france|england|america)\b",
        # Health (non-security)
        r"\bmedical advice\b",
        r"\bsymptoms of\b",
        r"\btreat(ment|ing) for\b",
        r"\bdoctor\b",
        r"\bprescription\b",
        # Shopping
        r"\bbest (phone|laptop|car|tv|headphones|camera)\b",
        r"\bproduct review\b",
        r"\bamazon\b",
        r"\bdiscount code\b",
        r"\bcoupon\b",
        # Real estate
        r"\bhouse price\b",
        r"\brent (apartment|house)\b",
        r"\bmortgage\b",
        # Coding (non-security)
        r"\bhello world\b",
        r"\bpython tutorial\b",
        r"\breact (tutorial|component)\b",
        r"\bjavascript tutorial\b",
        r"\bcss tutorial\b",
        # Vendor/tool-specific implementation details outside ISM guidance
        r"\b(cisco|juniper|asr\s*9000)\b",
        r"\bcisco asa\b",
        r"\bbgp route reflectors?\b",
        r"\bregistry keys?\b",
        r"\b(group policy object|gpo)\b",
        r"\bpython commands?\b",
        r"\btensorflow\b",
        r"\bbash script\b",
        r"\bkubernetes cluster\b",
        r"\bdocker desktop\b",
        r"\bjvm memory\b",
        r"\bapache tomcat\b",
        r"\bsource code for an exploit\b",
        r"\bexploit targeting\b",
        r"\blog4shell\b",
        r"\bnist sp 800-53\b",
        r"\biso/iec 27001\b",
        r"\bamerican healthcare provider\b",
        r"\bpricing differences\b",
        r"\bcost comparison\b",
        # Social media
        r"\binstagram\b",
        r"\btiktok\b",
        r"\btwitter\b",
        r"\bfacebook\b",
        r"\bsnapchat\b",
        # Politics
        r"\bwho won the election\b",
        r"\bpresident of\b",
        r"\bprime minister of\b",
        # Misc obvious off-topic
        r"\bpizza\b",
        r"\bburger\b",
        r"\bsushi\b",
        r"\bwhat is love\b",
        r"\bmeaning of life\b",
        r"\btranslate .+ to\b",
        r"\bconvert .+ to\b",
    ]
]

ALLOW_SIGNALS = [
    "ism",
    "security",
    "cyber",
    "encryption",
    "firewall",
    "access control",
    "authentication",
    "mfa",
    "multi-factor",
    "multi factor",
    "password",
    "passphrase",
    "vulnerability",
    "patch",
    "network",
    "audit",
    "compliance",
    "risk",
    "data protection",
    "incident",
    "malware",
    "phishing",
    "backup",
    "cloud security",
    "essential eight",
    "essential 8",
    "asd",
    "acsc",
    "gateway",
    "hardening",
    "logging",
    "monitoring",
    "privileged access",
    "cryptograph",
    "tls",
    "ssl",
    "vpn",
    "ipsec",
    "dns security",
    "intrusion",
    "endpoint",
    "antivirus",
    "data spill",
    "classification",
    "sanitisation",
    "sanitization",
    "disposal",
    "media destruction",
    "cable",
    "emanation",
    "tempest",
    "wireless",
    "bluetooth",
    "rfid",
    "byod",
    "mobile device",
    "removable media",
    "email security",
    "web filtering",
    "content filter",
    "cross domain",
    "virtualisation",
    "virtualization",
    "containerisation",
    "containerization",
    "database security",
    "system administration",
    "system management",
    "change management",
    "supply chain",
    "procurement",
    "personnel security",
    "physical security",
    "ict equipment",
    "server room",
    "data centre",
    "outsourcing",
    "information security",
    "protective marking",
    "need-to-know",
    "clearance",
    "authoris",
    "authoriz",
]


def pre_filter(question: str) -> tuple[bool, str]:
    if not OOS_PRE_FILTER_ENABLED:
        return (True, "disabled")

    for pattern in DENY_PATTERNS:
        if pattern.search(question):
            return (False, "off_topic")

    q_lower = question.lower()
    for signal in ALLOW_SIGNALS:
        if signal in q_lower:
            return (True, "topic_match")

    return (True, "uncertain")


def rerank_threshold_check(
    chunks: list[dict], threshold: float = OOS_RERANK_THRESHOLD
) -> tuple[bool, float]:
    if not chunks:
        return (False, 0.0)

    max_score = max(c.get("rerank_score", 0.0) for c in chunks)

    if max_score < threshold:
        return (False, max_score)

    return (True, max_score)
